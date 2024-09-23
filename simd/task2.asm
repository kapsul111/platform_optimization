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

section .data
    array_size equ 1024 * 1024

    element_size equ 4

    allocation_error_str db "Allocation error", 0xa, 0
    array_str db "Array output", 0xa, 0
    simd_time_str db "Simd time: %d", 0xa, 0
    loop_time_str db "Loop time: %d", 0xa, 0
    success_str db "Arrays are the same, calculation is correct",0xa, 0
    fail_str db "Arrays are not the same, calculation is wrong",0xa, 0
    number_str db "%i ", 0
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

;calculate simd result
    get_time start_time

    mov rsi, [first_array]
    mov rbx, [second_array]
    mov rcx, array_size
    mov rdi, [output_array_simd]

    call calculate_add_avx

    get_time end_time

    mov rax, [end_time+8]
    mov rbx, [start_time+8]
    sub rax, [start_time+8]

    mov rdi, simd_time_str
    mov esi, eax
    xor rax, rax
    call printf

;calculate loop result
    get_time start_time

    mov rsi, [first_array]
    mov rbx, [second_array]
    mov rcx, array_size
    mov rdi, [output_array_loop]

    call calculate_per_one_loop

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

calculate_add_avx:
    ;rsi - first_array
    ;rbx - second_array
    ;rcx - array_size
    ;rdi - output_array

    cmp rcx, 16
    jle calculate_per_one_loop

calculate_add_loop:
    vmovdqa64 zmm0, [rsi]
    vmovdqa64 zmm1, [rbx]
    vpaddd zmm2, zmm1, zmm0

    vmovdqa64 [rdi], zmm2

    add rsi, 64
    add rbx, 64
    add rdi, 64
    sub rcx, 16

    cmp rcx, 16
    jg calculate_add_loop

calculate_per_one_loop: 
    ;could be used as procedure with the same registers

    mov eax, [rsi]
    add eax, [rbx]
    mov dword [rdi], eax

    add rsi, 4
    add rbx, 4
    add rdi, 4

    loop calculate_per_one_loop

    ret

populate:
    ;rsi array
    ;rbx max value
    ;size is array_size

    push rax
    push rcx
    push rdx
    ;incrementing values

    mov rcx, array_size

populate_loop:
    xor rdx, rdx
    mov rax, rcx
    div rbx

    mov dword [rsi], edx
    add rsi, element_size

    loop populate_loop

    pop rdx
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
    mov esi, [r13]
    xor eax, eax
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

