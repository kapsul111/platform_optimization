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

   %macro write_newline 0
      write_string newline, newline_len
   %endmacro

   %macro write_array_macro 2
    mov esi, %1
    mov ecx, %2

    call write_array
    write_newline
   %endmacro
   
   section .data
    source db 1, 9, 7, 3, 5, 2, 8, 6, 4
    dest db 9 dup(0)
    newline db 0xa
    newline_len equ $ - newline

section .bss
    write_buf resb 1

section .text
    global _start

_start:
    write_array_macro source, 9
    
    mov ecx, 9
    mov esi, source
    mov edi, dest

copy_loop:
    mov al, [esi]
    mov [edi], al

    inc esi
    inc edi
    loop copy_loop

init_sort_loop:
    mov ecx, 8

    mov esi, 8
    mov edi, 7

    mov al, 1

sort_loop:
    mov bl, [dest + esi]
    mov bh, [dest + edi]

    cmp bl, bh
    jge next

    mov al, 0
    mov [dest + esi], bh
    mov [dest + edi], bl

    next:
    dec esi
    dec edi

    loop sort_loop

    test al, al
    jz init_sort_loop

    write_array_macro dest, 9
    
    mov eax, 1
    xor ebx, ebx
    int 0x80

write_array:
    write_loop:
        mov al, [esi]
        add al, '0'
        mov [write_buf], al

        write_string write_buf, 1

        inc esi
        loop write_loop
    ret



