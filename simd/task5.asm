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

    main_string resq 1
    substring resq 1

    result_simd resd 1
    result_loop resd 1

section .data
    main_str_size equ 1024 * 1024
    substr_size equ 1024

    allocation_error_str db "Allocation error", 0xa, 0
    array_str db "Array output", 0xa, 0
    avx_result_str db "Simd substrin search result position: %i", 0xa, 0
    loop_result_str db "Loop substrin search result position: %i", 0xa, 0

    simd_time_str db "Simd time: %d", 0xa, 0
    loop_time_str db "Loop time: %d", 0xa, 0

section .text
    global _start
    extern posix_memalign
    extern printf
    extern memcmp

_start:
;allocate
    mov rdi, main_string
    mov rsi, 64
    mov rdx, main_str_size
    
    call posix_memalign

    test rax, rax
    jnz allocation_error

    mov rdi, substring
    mov rsi, 64
    mov rdx, substr_size

    call posix_memalign

    test rax, rax
    jnz allocation_error
    
;populate strings
    mov rsi, [main_string]
    mov rdi, main_str_size
    mov rdx, substr_size
    call populate_main_string

    mov rsi, [substring]
    mov rdi, substr_size
    call populate_substring

; search substring with avx
    get_time start_time

    mov rsi, [main_string]
    mov rcx, main_str_size
    mov rdi, [substring]
    mov rdx, substr_size

    call avx_substring_search
    mov [result_simd], rax

    get_time end_time

    mov rax, [end_time+8]
    mov rbx, [start_time+8]
    sub rax, [start_time+8]

    mov rdi, simd_time_str
    mov esi, eax
    xor rax, rax
    call printf

;search substring with loop
    get_time start_time

    mov rsi, [main_string]
    mov rcx, main_str_size
    mov rdi, [substring]
    mov rdx, substr_size

    call substring_search

    mov [result_loop], rax

    get_time end_time

    mov rax, [end_time+8]
    mov rbx, [start_time+8]
    sub rax, [start_time+8]

    mov rdi, loop_time_str
    mov esi, eax
    xor rax, rax
    call printf

;output dot multiplication time
    mov rdi, avx_result_str
    mov rsi, [result_simd]
    mov eax, 0
    call printf

    mov rdi, loop_result_str
    mov rsi, [result_loop]
    mov eax, 0
    call printf

exit:
    mov rax, 60
    mov rdi, rdi
    syscall
 
allocation_error:
    mov rdi, allocation_error_str
    xor rax, rax
    call printf
    jmp exit

avx_substring_search:
    ; rsi, [main_string]
    ; rcx, [main_str_size]
    ; rdi, [substring]
    ; rdx, [substr_size]

    vpbroadcastb zmm15, [rdi]
    mov r8, 0

avx_loop:
    vmovaps zmm0, [rsi + r8]
    vpcmpeqb k1, zmm0, zmm15

avx_check_str:
    ktestw k1, k1
    jz avx_not_equal

    lzcnt rax, rbx

    mov r10, rax

    mov r13, rsi
    add r13, r8
    add r13, rax

    mov r14, rdi
    
    push rdi
    push rsi
    push rdx

    lea rdi, [r13]
    lea rsi, [r14]
    
    mov rdx, [rdx]
    
    call memcmp

    pop rdx
    pop rsi
    pop rdi

    cmp rax, 0
    je avx_equal

    ;k1 &= ~(1ULL << (64 - r10 - 1))
    mov rax, 64
    sub rax, r10
    dec rax

    push rcx

    mov al, cl
    mov r12, 1
    shl r12, cl

    not r12

    kmovq r11, k1
    and r11, r12
    kmovq k1, r11

    pop rcx

    jmp avx_check_str

avx_equal:
    mov rax, r8
    ret

avx_not_equal:
    add r8, 64
	
    ; rax = ((rcx - rdx + 1) / 64) * 64;
    mov rax, rcx
    sub rax, rdx
    inc rax
    mov r10, 64
    
    push rdx
    xor rdx, rdx

    div r10
    mul r10
    
    pop rdx

    cmp r8, rax
    jl avx_loop

jmp avx_substring_remainder

substring_search:
    ;could be used as procedure with the same registers

    mov r8, 0

avx_substring_remainder:
    mov r15, rcx
    sub r15, rdx
    inc r15

outer_substring_search_loop: 

    mov r11, 1
    mov r10, r8

    mov r9, 0
inner_substring_search_loop:

    mov rax, rsi
    add rax, r10
    mov r14b, [rax]

    cmp r14b, [rdi + r9]
    jne substring_not_match

    inc r10
    inc r9
    cmp r9, rdx
    jl inner_substring_search_loop
    
    mov rax, r8
    ret

substring_not_match:
    inc r8
    cmp r8, r15
    jl outer_substring_search_loop

    mov rax, -1
    ret

populate_main_string:
    ;rsi string
    ;rdi main_string_size
    ;rdx substring_size

    mov rcx, rdi
    sub rcx, rdx
     
    mov bl, 'a'
populate_main_str_loop:
    mov byte [rsi], bl
    inc rsi
    loop populate_main_str_loop

    mov rcx, rdx
    mov bl, 'c'
populate_last_part:
    mov byte [rsi], bl
    inc rsi
    loop populate_last_part
    ret

populate_substring:
    ;rsi string
    ;rdi string_size
    
    mov rcx, rdi
    mov bl, 'c'
populate_substring_loop:
    mov byte [rsi], bl
    inc rsi
    loop populate_substring_loop

    ret
