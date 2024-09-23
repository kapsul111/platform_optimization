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

    first_array resq 1
    second_array resq 1
    output_array_simd resq 1
    output_array_loop resq 1

    dot_multiple_result_simd resd 1
    dot_multiple_result_loop resd 1

section .data
    float_init_value dd 23.45

    array_size equ 1024 * 1024

    element_size equ 4

    allocation_error_str db "Allocation error", 0xa, 0
    array_str db "Array output", 0xa, 0
    avx_dot_output_str db "Loop dot multiplication output: %f", 0xa, 0
    loop_dot_output_str db "Avx dot multiplication output: %f", 0xa, 0

    simd_time_str db "Simd time: %d", 0xa, 0
    loop_time_str db "Loop time: %d", 0xa, 0

    dot_simd_time_str db "Avx dot multiplication time: %d", 0xa, 0
    dot_loop_time_str db "Loop dot multiplication time: %d", 0xa, 0

    success_str db "Arrays are the same, calculation is correct", 0xa, 0
    fail_str db "Arrays are not the same, calculation is wrong", 0xa, 0

    number_str db "%f ", 0
    newline_str db 10, 0

section .text
    global _start
    extern posix_memalign
    extern printf

_start:
;allocate
    mov rdi, first_array
    mov rsi, 64
    mov rdx, array_size * 4
    
    call posix_memalign

    test rax, rax
    jnz allocation_error

    mov rdi, second_array
    mov rsi, 64
    mov rdx, array_size * 4

    call posix_memalign

    test rax, rax
    jnz allocation_error
    
    mov rdi, output_array_simd
    mov rsi, 64
    mov rdx, array_size * 4

    call posix_memalign

    test rax, rax
    jnz allocation_error
    
    mov rdi, output_array_loop
    mov rsi, 64
    mov rdx, array_size * 4

    call posix_memalign

    test rax, rax
    jnz allocation_error

;populate arrays
    mov rsi, [first_array]
    mov rbx, 3
    call populate

    mov rsi, [second_array]
    mov rbx, 5
    call populate

;calculate sum simd result
    get_time start_time

    mov rsi, [first_array]
    mov rbx, [second_array]
    mov rcx, array_size
    mov rdi, [output_array_simd]

    call sum_avx

    get_time end_time

    mov rax, [end_time+8]
    mov rbx, [start_time+8]
    sub rax, [start_time+8]

    mov rdi, simd_time_str
    mov esi, eax
    xor rax, rax
    call printf

;calculate avx loop result
    get_time start_time

    mov rsi, [first_array]
    mov rbx, [second_array]
    mov rcx, array_size
    mov rdi, [output_array_loop]

    call sum_loop

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

    ; mov rdi, [first_array]
    ; mov rsi, array_size
    ; call write_array
    
    ; mov rdi, array_str
    ; call printf

    ; mov rdi, [second_array]
    ; mov rsi, array_size
    ; call write_array

    ; mov rdi, array_str
    ; xor rax, rax
    ; call printf

    ; mov rdi, [output_array_simd]
    ; mov rsi, array_size
    ; call write_array
    
    ; mov rdi, array_str
    ; xor rax, rax
    ; call printf
    
    ; mov rdi, [output_array_loop]
    ; mov rsi, array_size
    ; call write_array

;compare
    mov rdi, [output_array_loop]
    mov rsi, [output_array_simd]
    mov rcx, array_size
    call compare_arrays

    cmp rax, 0
    je failed

    mov rdi, success_str
    xor rax, rax
    call printf

;calculate dot multiplication avx

    get_time start_time

    mov rsi, [first_array]
    mov rbx, [second_array]
    mov rcx, array_size
    mov rdx, dot_multiple_result_simd

    call avx_dot_multiplication

    get_time end_time
    
    mov rax, [end_time+8]
    mov rbx, [start_time+8]
    sub rax, [start_time+8]

    mov rdi, dot_simd_time_str
    mov esi, eax
    xor rax, rax
    call printf


;calculate dot multiplication loop
    get_time start_time

    mov rsi, [first_array]
    mov rbx, [second_array]
    mov rcx, array_size
    mov rdx, dot_multiple_result_loop

    call loop_dot_multiplication

    get_time end_time
    
    mov rax, [end_time+8]
    mov rbx, [start_time+8]
    sub rax, [start_time+8]

    mov rdi, dot_loop_time_str
    mov esi, eax
    xor rax, rax
    call printf

;output dot multiplication time
    mov rdi, avx_dot_output_str
    movss xmm0, [dot_multiple_result_simd]
    cvtss2sd xmm0, xmm0
    mov eax, 1
    call printf

    vxorps xmm0, xmm0, xmm0
    mov rdi, loop_dot_output_str
    movss xmm0, [dot_multiple_result_loop]
    cvtss2sd xmm0, xmm0
    mov eax, 1
    call printf

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

sum_avx:
    ;rsi - first_array
    ;rbx - second_array
    ;rcx - array_size
    ;rdi - output_array

    cmp rcx, 16
    jle sum_loop

sum_avx_loop:
    vmovaps zmm0, [rsi]
    vmovaps zmm1, [rbx]
    vaddps zmm2, zmm1, zmm0

    vmovaps [rdi], zmm2

    add rsi, 64
    add rbx, 64
    add rdi, 64
    sub rcx, 16

    cmp rcx, 16
    jg sum_avx_loop

    vxorps xmm0, xmm0, xmm0

sum_loop: 
    ;could be used as procedure with the same registers

    movss xmm0, [rsi]
    addss xmm0, [rbx]

    movss [rdi], xmm0

    add rsi, 4
    add rbx, 4
    add rdi, 4

    loop sum_loop

    ret

avx_dot_multiplication:
    ;rsi - first_array
    ;rbx - second_array
    ;rcx - array_size
    ;rdx - output value

    vxorps zmm0, zmm0, zmm0
    vxorps zmm1, zmm1, zmm1
    vxorps zmm3, zmm3, zmm3

    cmp rcx, 16
    jle avx_dot_loop_remainer

avx_dot_loop:
    vmovaps zmm0, [rsi]
    vmovaps zmm1, [rbx]
    vmulps zmm2, zmm1, zmm0

    vaddps zmm3, zmm2

    add rsi, 64
    add rbx, 64
    sub rcx, 16

    cmp rcx, 16
    jg avx_dot_loop

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

avx_dot_loop_remainer:
    movss xmm0, [rsi]
    movss xmm1, [rbx]

    mulss xmm0, xmm1
    
    addss xmm7, xmm0
    loop avx_dot_loop_remainer
    
    movss dword [rdx], xmm7
    ret

loop_dot_multiplication:
    ;rsi - first_array
    ;rbx - second_array
    ;rcx - array_size
    ;rdx - output value

    vxorps xmm7, xmm7, xmm7

multiplication_loop:
    movss xmm0, [rsi]
    movss xmm1, [rbx]

    mulss xmm0, xmm1

    addss xmm7, xmm0

    add rsi, 4
    add rbx, 4

    loop multiplication_loop

    movss dword [rdx], xmm7
    ret

populate:
    ;rsi array
    ;rbx max value
    ;size is array_size

    push rax
    push rcx

    mov rcx, array_size

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

    add r13, 4
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

    add rsi, 4
    add rdi, 4
    
    loop compare_loop

    mov rax, 1
    ret

compare_not_equal:
    mov rax, 0
    ret

