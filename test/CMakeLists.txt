configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(PCOV_TEST_DEPENDS
  FileCheck
  not
  pcov)

add_lit_testsuite(check-pcov "Run pcov tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${PCOV_TEST_DEPENDS})
set_target_properties(check-pcov PROPERTIES FOLDER "Tests")
