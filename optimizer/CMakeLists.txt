cmake_minimum_required(VERSION 3.16)


add_library(optim INTERFACE)
target_link_libraries(optim INTERFACE nn)