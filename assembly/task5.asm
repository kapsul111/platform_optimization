section .data
    numbers db 1, 6, 3, 5, 2
    len equ $ - numbers

section .bss
    res resb 1

section .text
    global _start

_start:
    mov bl, [numbers]
    mov esi, numbers
    mov ecx, len

SIZE_CHECK:
    cmp ecx, 4
    jl LOOP_SMALL

LOOP_BIG:
    mov al, [esi]
    mov ah, [esi + 1]
    mov dl, [esi + 2]
    mov dh, [esi + 3]

    cmp bl, al
    jge CHECK2
    mov bl, al

    CHECK2:
    cmp bl, ah
    jge CHECK3
    mov bl, ah

    CHECK3:
    cmp bl, dl
    jge CHECK4
    mov bl, dl
    
    CHECK4:
    cmp bl, dh
    jge NEXT_BIG
    mov bl, dh

NEXT_BIG:
    add esi, 4
    sub ecx, 4
    jnz SIZE_CHECK

LOOP_SMALL:
    cmp bl, [esi]
    jge NEXT_SMALL
    mov bl, [esi]

NEXT_SMALL:
    inc esi
    dec ecx
    jnz SIZE_CHECK

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