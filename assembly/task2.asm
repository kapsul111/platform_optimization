segment .data
    msg db 'Hello World', 0xa
    len equ $ - msg

segment .bss
    rev_msg resb len

segment .text
    global _start

_start:
    mov edi, rev_msg
    mov esi, msg
    add esi, len
    dec esi

    mov ecx, len

    l1:
        mov al, [esi]

        cmp al, 97
        jl big

        sub al, 32

        big:

        mov [edi], al
        dec esi
        inc edi

    loop l1

    mov eax, 1
    int 0x80

