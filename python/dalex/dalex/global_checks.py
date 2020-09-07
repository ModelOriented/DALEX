from importlib import import_module
from re import search


def global_check_import(name=None, msg=""):
    # check if the optional dependency is available
    if name:
        try:
            import_module(name)
        except ImportError:
            print("Missing optional dependency '" + name + "'." +
                  msg +
                  "Use pip or conda to install " + name + ".")


def global_check_object_class(ob, class_object):
    # check if one of 'objects' is of the desired class - raise error if not
    if not isinstance(ob, class_object):
        class_name = search("(?<=<class ').*(?='>)", str(class_object))[0]
        raise TypeError("One of 'objects' isn't of the " + class_name + " class: " + str(type(ob)))


def global_raise_objects_class(ob, class_object):
    # raise error if 'objects' isn't of the desired class
    class_name = search("(?<=<class ').*(?='>)", str(class_object))[0]
    raise TypeError(
        "'objects' should be either " + class_name +
        " object or list/tuple of " + class_name + " objects: " + str(type(ob))
    )
