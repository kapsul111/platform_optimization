section .data
    numbers db 1, 1, 3, 5, 2
    len equ $ - numbers

section .bss
    res resb 1

section .text
    global _start

_start:
    mov bl, [numbers]
    mov esi, numbers
    mov eax, len

LOOP1:
    cmp bl, [esi]
    jge NEXT

    mov bl, [esi]

NEXT:
    inc esi
    dec eax
    jnz LOOP1

DONE:
    add bl, '0'
    mov byte [res], bl

    mov eax, 4
    mov ebx, 1
    mov ecx, res
    mov edx, 1
    int 0x80

    mov eax, 1
    int 0x80


    
    