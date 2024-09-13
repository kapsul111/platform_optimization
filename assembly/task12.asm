%macro write_string 2 
    push eax
    push ebx
    push ecx
    push edx

    mov eax, 4
    mov ebx, 1
    mov ecx, %1
    mov edx, %2
    int 80h
      
    pop edx
    pop ecx
    pop ebx
    pop eax
%endmacro

section .data
   array_size equ 4
   
   first_array dd 0
   second_array dd 0

   res_same_vals dd 0
   res_same_val_size dd 0
   
   res_first_diff dd 0
   res_first_diff_size dd 0
   
   res_second_diff dd 0
   res_second_diff_size dd 0

   hello db "hello!", 0xa
   hello_len equ $ - hello

   test1 db "test!", 0xa
   test_len equ $ - test1

section .bss
    output_buf resb 16

section .text
   global _start
	
_start:
;allocate first array memory
    mov eax, array_size
    mov ebx, 4
    mul ebx
    call allocate    
    mov dword [first_array], eax

;allocate second array memory
    mov eax, array_size
    mov ebx, 4
    mul ebx
    call allocate    
    mov dword [second_array], eax

;populate first
    mov esi, [first_array]
    mov dword [esi], 1
    mov dword [esi + 4], 2
    mov dword [esi + 8], 3
    mov dword [esi + 12], 4

;populate second
    mov esi, [second_array]
    mov dword [esi], 5
    mov dword [esi + 4], 6
    mov dword [esi + 8], 3
    mov dword [esi + 12], 4

;allocate common array
    mov eax, array_size
    mov ebx, 4
    mul ebx
    call allocate
    
    mov dword [res_same_vals], eax

;allocate first unique elements array
    mov eax, array_size
    mov ebx, 4
    mul ebx
    call allocate
    
    mov dword [res_first_diff], eax

;allocate second unique elements array
    mov eax, array_size
    mov ebx, 4
    mul ebx
    call allocate
    
    mov dword [res_second_diff], eax

;find_same_elements
    call find_same_elements
    
;print same elements
    mov ebx, [res_same_val_size] 
    mov eax, [res_same_vals]
    call write_array

;find unique first array elements
    mov esi, [first_array]
    mov edi, [second_array]
    mov eax, [res_first_diff]
    call find_diff_elements
    mov dword [res_first_diff_size], eax

;print unique first array elements
    mov ebx, [res_first_diff_size] 
    mov eax, [res_first_diff]
    call write_array

;find unique second array elements
    mov esi, [second_array]
    mov edi, [first_array]
    mov eax, [res_second_diff]
    call find_diff_elements
    mov dword [res_second_diff_size], eax

;print unique second array elements
    mov ebx, [res_second_diff_size] 
    mov eax, [res_second_diff]
    call write_array

    mov eax, 16
    call free
    
    mov eax, 16
    call free
    
    mov eax, 16
    call free
    
    mov eax, 16
    call free
    
    mov eax, 16
    call free

exit:
    mov eax, 1
    xor ebx, ebx
    int 80h

allocate:
    ;eax - byte count
    push edx
    push ecx
    push ebx

    mov edx, eax ;save needed size
    
    mov eax, 45  ;find break
    xor ebx, ebx
    int 80h

    mov ecx, eax

    cmp eax, 0
    jl  exit
    
    add eax, edx
    mov ebx, eax
    mov eax, 45
    int 80h

    cmp eax, 0
    jl  exit

    mov eax, ecx
    
    pop ebx
    pop ecx
    pop edx

    ret

free:
    push edx
    push ecx
    push ebx

    mov edx, eax ;save needed size
    
    mov eax, 45  ;find break
    xor ebx, ebx
    int 80h

    mov ecx, eax

    cmp eax, 0
    jl  exit
    
    sub eax, edx
    mov ebx, eax
    mov eax, 45
    int 80h

    cmp eax, 0
    jl  exit

    mov eax, ecx
    
    pop ebx
    pop ecx
    pop edx

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

find_same_elements:
    mov esi, [first_array]
    mov edi, [second_array]
    mov eax, [res_same_vals]

    mov ecx, array_size
    xor ebx, ebx

outer_same_loop:
    push ecx
    mov edx, [esi]
    mov ecx, array_size

inner_same_loop:
    cmp edx, [edi]
    jne next_same_inner

    mov dword [eax], edx
    add eax, 4
    inc ebx
    jmp end_same_inner_loop

next_same_inner:
    add edi, 4
    loop inner_same_loop

end_same_inner_loop:
    add esi, 4
    mov edi, [second_array]
    pop ecx

    loop outer_same_loop

    mov dword [res_same_val_size], ebx

    ret


find_diff_elements:
    mov ecx, array_size
    xor ebx, ebx

outer_diff_loop:
    push ecx
    push edi
    mov edx, [esi]
    mov ecx, array_size

inner_diff_loop:
    cmp edx, [edi]
    je end_diff_inner_loop

    add edi, 4
    loop inner_diff_loop

    mov dword [eax], edx
    add eax, 4
    inc ebx

end_diff_inner_loop:
    add esi, 4
    pop edi
    pop ecx

    loop outer_diff_loop

    mov eax, ebx

    ret 