%macro write_string 2 
    push eax
    push ebx
    push ecx
    push edx

    mov ecx, %1
    mov edx, %2
    mov eax, 4
    mov ebx, 1
    int 80h
      
    pop edx
    pop ecx
    pop ebx
    pop eax
%endmacro

%macro get_time 1 
    push eax
    push ebx
    push ecx
    push edx

    mov eax, 78 
    mov ebx, %1
    xor ecx, ecx
    int 80h
      
    pop edx
    pop ecx
    pop ebx
    pop eax
%endmacro

%macro int_to_str 2
    mov esi, %1
    mov ecx, %2
    dec ecx

    call int_to_str_pr 
%endmacro

section .bss
    output_buf resb 1024
    start_time resd 2
    end_time resd 2

    simd_time_buf resb 4

section .data
    array_size equ 1048576
    first_array dd 0
    second_array dd 0
    output_array_simd dd 0
    output_array_loop dd 0
    base_address dd 0

    element_size equ 4

    allocation_error db "Allocation error", 0xa
    allocation_error_len equ $ - allocation_error
    
    array_str db "Array output", 0xa
    array_str_len equ $ - array_str
    
    simd_time_str db "Simd time:"
    simd_time_str_len equ $ - simd_time_str
    
    loop_time_str db "Loop time:"
    loop_time_str_len equ $ - loop_time_str

    success_str db "Arrays are the same, calculation is correct",0xa
    success_str_len equ $ - success_str

    fail_str db "Arrays are not the same, calculation is wrong",0xa
    fail_str_len equ $ - fail_str    

section .text
    global _start

_start:
;find base_address for free later
    mov eax, 45  ;find break
    xor ebx, ebx
    int 80h

    mov dword [base_address], eax
    
;allocate first array memory
    mov eax, array_size
    mov ebx, element_size
    mul ebx
    mov ebx, 64
    call allocate    
    mov dword [first_array], eax
    
;allocate second array memory
    mov eax, array_size
    mov ebx, element_size
    mul ebx
    mov ebx, 64
    call allocate    
    mov dword [second_array], eax
    
;allocate simd output array for memory
    mov eax, array_size
    mov ebx, element_size
    mul ebx
    mov ebx, 64
    call allocate    
    mov dword [output_array_simd], eax

;allocate loop output array  memory
    mov eax, array_size
    mov ebx, element_size
    mul ebx
    mov ebx, 64
    call allocate    
    mov dword [output_array_loop], eax

;populate value
    mov esi, [first_array]
    mov ebx, 3
    call populate

    mov esi, [second_array]
    mov ebx, 5
    call populate

;calculate simd result
    get_time start_time

    mov esi, [first_array]
    mov ebx, [second_array]
    mov ecx, array_size
    mov edi, [output_array_simd]

    call calculate_add_avx

    get_time end_time

    mov eax, [end_time+4]
    mov ebx, [start_time+4]
    sub eax, [start_time+4]

    write_string simd_time_str, simd_time_str_len 

    int_to_str simd_time_buf, 8
    write_string simd_time_buf, 8

;calculate loop result
    get_time start_time

    mov esi, [first_array]
    mov ebx, [second_array]
    mov ecx, array_size
    mov edi, [output_array_loop]

    call calculate_per_one_loop

    get_time end_time

    mov eax, [end_time+4]
    mov ebx, [start_time+4]
    sub eax, [start_time+4]

    write_string loop_time_str, loop_time_str_len 

    int_to_str simd_time_buf, 8
    write_string simd_time_buf, 8

;output arrays
    ; write_string array_str, array_str_len

    ; mov eax, [first_array]
    ; mov ebx, array_size
    ; call write_array
    
    ; write_string array_str, array_str_len

    ; mov eax, [second_array]
    ; mov ebx, array_size
    ; call write_array

    ; write_string array_str, array_str_len

    ; mov eax, [output_array_simd]
    ; mov ebx, array_size
    ; call write_array
    
    ; write_string array_str, array_str_len
    
    ; mov eax, [output_array_loop]
    ; mov ebx, array_size
    ; call write_array

    mov esi, [output_array_simd]
    mov edi, [output_array_loop]
    mov ecx, array_size
    call compare_arrays

    cmp eax, 0
    je failed

    write_string success_str, success_str_len

exit:
    mov eax, 1
    int 0x80
 
failed:
    write_string fail_str, fail_str_len

    jmp exit

error:
    write_string allocation_error, allocation_error_len
    jmp exit

calculate_add_avx:
    ;esi - first_array
    ;ebx - second_array
    ;ecx - array_size
    ;edi - output_array

    cmp ecx, 16
    jle calculate_per_one_loop

calculate_add_loop:
    vmovaps zmm0, [esi]
    vmovaps zmm1, [ebx]
    vaddps zmm2, zmm1, zmm0

    vmovaps [edi], zmm2

    add esi, 64
    add ebx, 64
    add edi, 64
    sub ecx, 16

    cmp ecx, 16
    jg calculate_add_loop

calculate_per_one_loop: 
    ;could be used as procedure with the same registers

    mov eax, [esi]
    add eax, [ebx]
    mov dword [edi], eax

    add esi, 4
    add ebx, 4
    add edi, 4

    loop calculate_per_one_loop

    ret

allocate:
    ;eax - byte count
    ;ebx - needed aligment
    push edi
    push ecx
    push ebx

    mov edi, eax ;save needed size
    
    mov eax, 45  ;find break
    xor ebx, ebx
    int 80h

    pop ebx

    ;make proper alignment (ebx argument)
    call find_aligned_address
    ;done with finding aligment

    mov ecx, eax

    cmp eax, 0
    jl  error
    
    add eax, edi
    mov ebx, eax
    mov eax, 45
    int 80h

    cmp eax, 0
    jl  error

    mov eax, ecx
    
    pop ecx
    pop edi

    ret

find_aligned_address:
    ; eax - base address
    ; ebx - needed alignment

    ;aligned_address = (address + alignment - 1) & ~(alignment - 1)

    dec ebx 
    add eax, ebx
    not ebx 
    and eax, ebx 
    not ebx

    ret

populate:
    ;esi array
    ;ebx max value
    ;size is array_size

    push eax
    push ecx
    push edx
    ;incrementing values

    mov ecx, array_size

populate_loop:
    xor edx, edx
    mov eax, ecx
    div ebx

    mov dword [esi], edx
    add esi, element_size

    loop populate_loop

    pop edx
    pop ecx
    pop eax

    ret

write_array:
    ;eax - array ptr
    ;ebx - count (4byte each)

    mov ecx, ebx
    mov esi, eax
    mov edi, output_buf

    write_array_loop:
        mov eax, [esi]
        add eax, '0'
        mov byte [edi], al
        add esi, 4
        inc edi
    loop write_array_loop   

    mov byte [edi], 0xa
    mov eax, edi
    sub eax, output_buf
    inc eax

    write_string output_buf, eax 
    ret

int_to_str_pr:
    add esi, ecx
    mov ebx, 10
    int_to_str_loop:
        cmp eax, 0
        je int_to_str_done
        mov edx, 0
        div ebx
        add dl, '0'
        mov byte [esi], dl
        dec esi
    loop int_to_str_loop

    inc esi
    mov al, 0xA
    mov byte [esi], al

    int_to_str_done:
    ret   

compare_arrays:
    ; esi - first array
    ; edi - second array
    ; ecx - arrays size

compare_loop:
    mov eax, [esi]
    mov ebx, [edi]
    cmp eax, ebx
    jne compare_not_equal

    add esi, 4
    add edi, 4
    
    loop compare_loop

    mov eax, 1
    ret

compare_not_equal:
    mov eax, 0
    ret

