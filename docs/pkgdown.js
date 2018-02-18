$(function() {
  $("#sidebar").stick_in_parent({offset_top: 40});
  $('body').scrollspy({
    target: '#sidebar',
    offset: 60
  });

  var cur_path = location.href;
  $("#navbar ul li a").each(function(index, value) {
    if (value.text == "Home")
      return;
    if (value.getAttribute("href") === "#")
      return;

	var path = value.href;
    if (cur_path == path) {
      // Add class to parent <li>, and enclosing <li> if in dropdown
      var menu_anchor = $(value);
      menu_anchor.parent().addClass("active");
      menu_anchor.closest("li.dropdown").addClass("active");
    }
  });
});
