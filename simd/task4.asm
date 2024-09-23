%macro get_time 1
    push rax
    push rbx
    push rcx
    push rdx

    mov rax, 96
    mov rdi, %1
    xor rsi, rsi
    syscall

    pop rdx
    pop rcx
    pop rbx
    pop rax
%endmacro

section .bss
    align 16
    start_time resq 2
    align 16
    end_time resq 2

    first_matrix resq 1
    second_matrix resq 1
    transposed_second_matrix resq 1

    result_simd resq 1
    result_loop resq 1

section .data
    float_init_value dd 1.2345

    row_size equ 512
    matrix_size equ 262144

    element_size equ 4

    allocation_error_str db "Allocation error", 0xa, 0
    array_str db "Array output", 0xa, 0
    avx_output_str db "Loop matrix multiplication output: %f", 0xa, 0
    loop_output_str db "Avx matrix multiplication output: %f", 0xa, 0

    simd_time_str db "Simd time: %d", 0xa, 0
    loop_time_str db "Loop time: %d", 0xa, 0

    success_str db "Matrixes are the same, calculation is correct", 0xa, 0
    fail_str db "Matrixes are not the same, calculation is wrong", 0xa, 0

    number_str db "%f ", 0
    newline_str db 10, 0

section .text
    global _start
    extern posix_memalign
    extern printf

_start:
;allocate
    mov rdi, first_matrix
    mov rsi, 64
    mov rdx, matrix_size * 4
    
    call posix_memalign

    test rax, rax
    jnz allocation_error

    mov rdi, second_matrix
    mov rsi, 64
    mov rdx, matrix_size * element_size

    call posix_memalign

    mov rdi, transposed_second_matrix
    mov rsi, 64
    mov rdx, matrix_size * element_size

    call posix_memalign

    test rax, rax
    jnz allocation_error
    
    mov rdi, result_simd
    mov rsi, 64
    mov rdx, matrix_size * element_size

    call posix_memalign

    test rax, rax
    jnz allocation_error
    
    mov rdi, result_loop
    mov rsi, 64
    mov rdx, matrix_size * element_size

    call posix_memalign

    test rax, rax
    jnz allocation_error

;populate arrays
    mov rsi, [first_matrix]
    mov rbx, 3
    call populate

    mov rsi, [second_matrix]
    mov rbx, 5
    call populate

;transpose second matrix
    mov rsi, [second_matrix]
    mov rdi, [transposed_second_matrix]
    mov rcx, row_size
    call transpose_matrix

;calculate matrix multiplication avx
    get_time start_time

    mov rsi, [first_matrix]
    mov rbx, [transposed_second_matrix]
    mov rcx, row_size
    mov rdx, [result_simd]

    call avx_matrix_multiplication

    get_time end_time
    
    mov rax, [end_time+8]
    mov rbx, [start_time+8]
    sub rax, [start_time+8]

    mov rdi, simd_time_str
    mov esi, eax
    xor rax, rax
    call printf

;calculate matrix multiplication loop
    get_time start_time

    mov rsi, [first_matrix]
    mov rbx, [transposed_second_matrix]
    mov rcx, row_size
    mov rdx, [result_loop]

    call loop_matrix_multiplication

    get_time end_time
    
    mov rax, [end_time+8]
    mov rbx, [start_time+8]
    sub rax, [start_time+8]

    mov rdi, loop_time_str
    mov esi, eax
    xor rax, rax
    call printf

; output arrays
    ; mov rdi, array_str
    ; xor rax, rax
    ; call printf

    ; mov rdi, [first_matrix]
    ; mov rsi, matrix_size
    ; call write_array
    
    ; mov rdi, array_str
    ; call printf

    ; mov rdi, [transposed_second_matrix]
    ; mov rsi, matrix_size
    ; call write_array

    ; mov rdi, array_str
    ; xor rax, rax
    ; call printf

    ; mov rdi, [result_simd]
    ; mov rsi, matrix_size
    ; call write_array
    
    ; mov rdi, array_str
    ; xor rax, rax
    ; call printf
    
    ; mov rdi, [result_loop]
    ; mov rsi, matrix_size
    ; call write_array

exit:
    mov rax, 60
    mov rdi, rdi
    syscall
 
failed:
    mov rdi, fail_str
    xor rax, rax
    call printf

    jmp exit

allocation_error:
    mov rdi, allocation_error_str
    xor rax, rax
    call printf
    jmp exit

avx_matrix_multiplication:
    ;rsi - first_array
    ;rbx - second_array
    ;rcx - array_size
    ;rdx - output value

    mov r8, 0
avx_x_multiplication_loop:

    mov r9, 0
avx_y_multiplication_loop:
    vxorps xmm7, xmm7, xmm7
    
    mov r10, 0

    cmp rcx, 16
    jle avx_loop_remainer

avx_inside_multiplication_loop:

    vxorps zmm0, zmm0, zmm0
    vxorps zmm1, zmm1, zmm1
    vxorps zmm3, zmm3, zmm3

    mov rax, r8
    imul rax, rcx
    add rax, r10
    imul rax, element_size
    add rax, rsi
    vmovaps zmm0, [rax] ;take elements from first matrix

    mov rax, r9
    imul rax, rcx
    add rax, r10
    imul rax, element_size
    add rax, rbx
    vmovaps zmm1, [rax] ;take elements from second matrix

    vmulps zmm2, zmm1, zmm0

    vaddps zmm3, zmm2

    add r10, 16
    mov rax, rcx
    sub rax, r10

    cmp rax, 16
    jg avx_inside_multiplication_loop

    vextractf32x4 xmm4, zmm3, 0
    vextractf32x4 xmm5, zmm3, 1
    vextractf32x4 xmm6, zmm3, 2
    vextractf32x4 xmm7, zmm3, 3

    haddps xmm4, xmm4
    haddps xmm4, xmm4
    haddps xmm5, xmm5
    haddps xmm5, xmm5
    haddps xmm6, xmm6
    haddps xmm6, xmm6
    haddps xmm7, xmm7
    haddps xmm7, xmm7

    vaddps xmm4, xmm4, xmm5
    vaddps xmm6, xmm6, xmm7

    vxorps xmm7, xmm7, xmm7
    vaddps xmm7, xmm4, xmm6 

    vxorps xmm0, xmm0, xmm0
    vxorps xmm1, xmm1, xmm1

avx_loop_remainer: ;check if needed
    mov rax, r8
    imul rax, rcx
    add rax, r10
    imul rax, element_size
    add rax, rsi
    movss xmm0, [rax] ;take element from first matrix

    mov rax, r9
    imul rax, rcx
    add rax, r10
    imul rax, element_size
    add rax, rbx
    movss xmm1, [rax] ;take element from second matrix

    mulss xmm0, xmm1
    
    addss xmm7, xmm0
    inc r10
    cmp r10, rcx
    jl avx_loop_remainer
    
    mov rax, r8
    imul rax, rcx
    add rax, r9
    imul rax, element_size
    add rax, rdx
    movss dword [rax], xmm7

    inc r9
    cmp r9, rcx
    jl avx_y_multiplication_loop
    
    inc r8
    cmp r8, rcx
    jl avx_x_multiplication_loop

    ret

loop_matrix_multiplication:
    ;rsi - first_matrix
    ;rbx - second_matrix
    ;rcx - array_size
    ;rdx - output matrix

    mov r8, 0
x_multiplication_loop:

    mov r9, 0
y_multiplication_loop:
    vxorps xmm7, xmm7, xmm7
    
    mov r10, 0
inside_multiplication_loop:

    mov rax, r8
    imul rax, rcx
    add rax, r10
    imul rax, element_size
    add rax, rsi

    movss xmm0, [rax] ;take element from first matrix

    mov rax, r9
    imul rax, rcx
    add rax, r10
    imul rax, element_size
    add rax, rbx

    movss xmm1, [rax] ;take element from second matrix

    mulss xmm0, xmm1

    addss xmm7, xmm0

    inc r10
    cmp r10, rcx
    jl inside_multiplication_loop

    mov rax, r8
    imul rax, rcx
    add rax, r9
    imul rax, element_size
    add rax, rdx

    movss dword [rax], xmm7

    inc r9
    cmp r9, rcx
    jl y_multiplication_loop
    
    inc r8
    cmp r8, rcx
    jl x_multiplication_loop

    ret

populate:
    ;rsi array
    ;rbx max value
    ;size is array_size

    push rax
    push rcx

    mov rcx, row_size * row_size

    mov eax, [float_init_value]

populate_loop:
    mov dword [rsi], eax
    add rsi, 4

    loop populate_loop

    pop rcx
    pop rax

    ret

write_array:
    ;rdi - array ptr
    ;rsi - count (4byte each)
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    sub rsp, 8  ; Ensure 16-byte stack alignment

    mov r13, rdi
    mov r12, rsi

write_array_loop:
    mov rdi, number_str
    vxorps xmm0, xmm0, xmm0
    movss xmm0, [r13]
    cvtss2sd xmm0, xmm0
    mov eax, 1
    call printf

    add r13, element_size
    dec r12
    test r12, r12
    jnz write_array_loop

    mov rdi, newline_str
    call printf

    add rsp, 8
    pop r14
    pop r13
    pop r12
    pop rbp
    ret

compare_arrays:
    ; rsi - first array
    ; rdi - second array
    ; rcx - arrays size

compare_loop:
    mov eax, [rsi]
    mov ebx, [rdi]
    cmp eax, ebx
    jne compare_not_equal

    add rsi, element_size
    add rdi, element_size
    
    loop compare_loop

    mov rax, 1
    ret

compare_not_equal:
    mov rax, 0
    ret

transpose_matrix:
    ;rsi - input matrix
    ;rdi - output matrix
    ;rcx - row size

    xor r8, r8
x_transpose_loop:

    xor r9, r9
y_transpose_loop:
    ; rsi[(r8 * rcx + r9) * element_size]
    mov rax, r8
    imul rax, rcx
    add rax, r9
    imul rax, element_size
    add rax, rsi
    
    movss xmm0, [rax] ;take element from first matrix

    ; rdi[(r9 * rcx + r8) * element_size]
    mov rax, r9
    imul rax, rcx
    add rax, r8
    imul rax, element_size
    add rax, rdi

    movss dword [rax], xmm0

    inc r9
    cmp r9, rcx
    jl y_transpose_loop

    inc r8
    cmp r8, rcx
    jl x_transpose_loop

    ret