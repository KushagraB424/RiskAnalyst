find_package(GTest REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)

set(TEST_SOURCES
    test_finance.cpp
    test_statistics.cpp
    test_cuda.cpp
)

add_executable(RiskAnalystTests ${TEST_SOURCES})

target_link_libraries(RiskAnalystTests PRIVATE 
    GTest::gtest_main
    RiskAnalystLib
)

add_test(NAME RiskAnalystTests COMMAND RiskAnalystTests)
