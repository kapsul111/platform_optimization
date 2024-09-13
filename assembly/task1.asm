section .data
    var1 db 1
    var2 db 2
    var3 db 3

segment .bss
    res resb 1

section .text
    global _start

_start:
    mov al, [var1]
    mov cl, [var2]
    mov dl, [var3]
    add al, cl
    add al, dl
    add al, '0'
	mov byte [res], al

    mov eax, 1
    int 0x80


