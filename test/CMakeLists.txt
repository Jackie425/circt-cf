configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(CIRCT_CFA_TRACE_TEST_DEPENDS
  FileCheck
  not
  circt-cfa-trace)

add_lit_testsuite(check-circt-cfa-trace "Run circt-cfa-trace tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIRCT_CFA_TRACE_TEST_DEPENDS})
set_target_properties(check-circt-cfa-trace PROPERTIES FOLDER "Tests")
