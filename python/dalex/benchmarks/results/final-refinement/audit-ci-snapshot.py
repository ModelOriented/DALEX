"""Audit exact-revision migration evidence downloaded from GitHub Actions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

p=argparse.ArgumentParser()
for name in ['run','head','artifacts','output']:
    p.add_argument('--'+name,required=True)
a=p.parse_args()
root=Path.cwd()
artifacts=Path(a.artifacts)
read=lambda path:json.loads(path.read_text(encoding='utf-8-sig'))
ci=json.loads(subprocess.check_output(['gh','run','view',a.run,'--repo','brunocarlin/DALEX',
    '--json','headSha,status,conclusion,jobs,url'],text=True))
assert ci['headSha']==a.head and ci['status']=='completed' and ci['conclusion']=='success'
assert len(ci['jobs'])==23 and all(job['conclusion']=='success' for job in ci['jobs'])
source=root/'python/dalex/dalex'
files=sorted(source.rglob('*.py'))
digest=hashlib.sha256()
for path in files:
    digest.update(path.relative_to(source).as_posix().encode()+b'\0')
    digest.update(path.read_bytes().replace(b'\r\n',b'\n')+b'\0')
identity={'python_modules':len(files),'python_source_sha256':digest.hexdigest(),
    'normalization':'sorted relative paths and source bytes, CRLF normalized to LF'}
counts={'full':0,'neural':0,'pps':0,'legacy-estimator':0}
junit={}
float32_max={'pandas':0.,'polars':0.,'pyarrow':0.}
skip_reasons=set()
for path in artifacts.rglob('*.xml'):
    xml=ET.parse(path).getroot()
    suites=list(xml.iter('testsuite'))
    stats={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ['tests','errors','failures','skipped']}
    stats['passed']=stats['tests']-stats['errors']-stats['failures']-stats['skipped']
    kind={'neural-tests.xml':'neural','legacy-pps-tests.xml':'pps',
          'legacy-estimator-bridge.xml':'legacy-estimator'}.get(path.name,'full')
    assert stats['errors']==stats['failures']==0,(path,stats)
    assert (stats['passed'],stats['skipped'])=={'full':(2071,18),'neural':(8,0),
        'pps':(18,0),'legacy-estimator':(4,0)}[kind],(path,stats)
    counts[kind]+=1
    if kind=='full':
        properties=[v for v in xml.iter('property') if v.get('name','').startswith('float32-loss-max-absolute-error-')]
        assert len(properties)==54,(path,len(properties))
        for prop in properties:
            backend=prop.get('name').split('absolute-error-',1)[1].split('-',1)[0]
            float32_max[backend]=max(float32_max[backend],float(prop.get('value')))
        for case in xml.iter('testcase'):
            skip=case.find('skipped')
            if skip is not None:
                assert 'test_aspect' in case.get('classname',''),case.attrib
                skip_reasons.add(skip.get('message',''))
    junit[str(path.relative_to(artifacts))]=stats
assert counts=={'full':15,'neural':2,'pps':1,'legacy-estimator':1},counts
assert float32_max['pandas']==0,float32_max
ft=artifacts/'tests-3.14t-ubuntu-latest'
runtime=(ft/'runtime-state.log').read_text(encoding='utf-8')
states=[line for line in runtime.splitlines() if 'GIL enabled after' in line]
assert len(states)==21 and all(line.endswith('False') for line in states)
assert 'Py_GIL_DISABLED: True' in runtime and 'GIL enabled at runtime check: False' in runtime
probe=read(ft/'gil-import-probe.json')
assert len(probe)==21 and all(item['returncode']==0 and not item['gil_before'] for item in probe)
benches=list(artifacts.rglob('dalex-narwhals-benchmark*.json'))
assert len(benches)==6
for path in benches:
    value=read(path)
    assert value['library_source']==identity,(path,value['library_source'],identity)
    assert all(item['max_absolute_parity_error']==0 for item in value['datasets'])
compat=artifacts/'upstream-compatibility'
numeric,pickles,plots,helpers,labels=[read(compat/name) for name in [
    'upstream-parity.json','upstream-pickle-compatibility.json','upstream-plot-parity.json',
    'upstream-adapter-parity.json','upstream-labels-pandas3.json']]
for value in [numeric,plots,helpers,labels,pickles]:
    assert value['passed'] and value['current_source']==identity,value
assert numeric['numeric_outputs_compared']==29 and pickles['objects_loaded']==16
assert pickles['legacy_duplicate_pandas_exports']
assert plots['figure_count']==178 and helpers['output_count']==265 and labels['output_count']==129
exceptions=plots['triplot_hover_density_exception']+plots['dendrogram_hover_density_exception']
assert len(exceptions)==10
wheel=read(artifacts/'pandas-only-package/pandas-only-package.json')
assert wheel['passed'] and wheel['current_source']==identity and wheel['base_dendrogram']
assert len(wheel['families'])==12 and not any(wheel['optional_modules_present'].values())
assert wheel['core_dependency_bounds']['pandas']=='>=1.5.3' and wheel['core_dependency_bounds']['narwhals']=='>=2.0'
pps=read(artifacts/'minimum-pandas/pps-backend-parity.json')
assert pps['passed'] and pps['native_matrix_storage'] and pps['max_absolute_difference']==0
assert pps['backends']==['pandas','polars','pyarrow']
for folder,name,versions in [
    ('minimum-pandas','minimum-versions.txt',['numpy==1.23.5','pandas==1.5.3','narwhals==2.0.0']),
    ('minimum-polars','minimum-polars-versions.txt',['plotly==6.0.0','polars==1.0.0','narwhals==2.0.0']),
    ('legacy-estimator-bridge','legacy-estimator-versions.txt',['lightgbm==4.6.0','numpy==1.23.5','pandas==1.5.3'])]:
    actual=(artifacts/folder/name).read_text(encoding='utf-8').lower().splitlines()
    assert all(version in actual for version in versions),(folder,versions)
result={'head':a.head,'url':ci['url'],'jobs':23,
    'job_results':[{key:job[key] for key in ['name','conclusion']} for job in ci['jobs']],
    'library_git_tree':subprocess.check_output(['git','rev-parse',a.head+':python/dalex/dalex'],text=True).strip(),
    'test_git_tree':subprocess.check_output(['git','rev-parse',a.head+':python/dalex/test'],text=True).strip(),
    'source_identity':identity,'junit':junit,'junit_kinds':counts,'skip_reasons':sorted(skip_reasons),
    'float32_loss_instrumented_cases_per_full_suite':54,'float32_loss_max_absolute_error':float32_max,
    'float32_loss_native_tolerance':{'rtol':8*2**-23,'atol':2**-23},
    'runtime_import_checks':21,'gil_enabled_after_imports':False,
    'automatic_gil_fallback':[item['module'] for item in probe if item['gil_after']],
    'ci_benchmark_reports':6,'ci_benchmark_max_absolute_error':0,
    'numeric_outputs':29,'numeric_max_absolute_error':max(numeric['max_absolute_error_by_output'].values()),
    'legacy_helper_public_text_outputs':265,'pandas3_label_outputs':129,'legacy_pickle_objects':16,
    'legacy_duplicate_pandas_exports':True,
    'pandas_only_wheel_families':12,'pandas_only_base_dendrogram':True,
    'pps_native_matrix_storage':True,'upstream_figures':178,'exact_figure_payloads':168,
    'hover_density_exceptions':exceptions}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value for key,value in result.items() if key not in ['junit','job_results']},indent=2))
