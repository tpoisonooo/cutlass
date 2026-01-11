include_directories(
  .
)

add_custom_target(13_fused_two_gemms)

add_custom_target(13_fused_two_convs)

add_custom_target(13_two_tensor_op_fusion
DEPENDS 13_fused_two_gemms
        13_fused_two_convs
)

foreach(FUSION_CONV_EXAMPLE
  fused_two_convs_f16_sm90_rf
  fused_two_convs_f16_sm90_shmem
)

  cutlass_example_add_executable(
    13_${FUSION_CONV_EXAMPLE}
    ${FUSION_CONV_EXAMPLE}.cu
  )

  add_dependencies(13_fused_two_convs 13_${FUSION_CONV_EXAMPLE})

endforeach()

foreach(FUSION_GEMM_EXAMPLE
  fused_two_gemms_f16_sm90_rf
  fused_two_gemms_f16_sm90_shmem
)
  cutlass_example_add_executable(
    13_${FUSION_GEMM_EXAMPLE}
    ${FUSION_GEMM_EXAMPLE}.cu
  )

  add_dependencies(13_fused_two_gemms 13_${FUSION_GEMM_EXAMPLE})

endforeach()
