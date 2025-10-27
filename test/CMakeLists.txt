configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(CIRCT_CF_TEST_DEPENDS
  FileCheck
  not
  circt-cf)

add_lit_testsuite(check-circt-cf "Run circt-cf tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_CF_TEST_DEPENDS})
set_target_properties(check-circt-cf PROPERTIES FOLDER "Tests")
