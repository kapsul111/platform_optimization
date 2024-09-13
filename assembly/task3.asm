segment .data
    var1 db 6
    var2 db 7
    var3 db 1

segment .bss
    res resb 1

section .text
    global _start

_start: 
    mov AL, [var1]
    mov BL, [var2]
    mov CL, [var3]

    cmp AL, BL
    JG NEXT
    mov AL, BL
    
NEXT:
    cmp AL, CL
    JG DONE

    mov AL, CL

DONE:
    mov [res], AL
    add byte [res], '0'

    mov eax, 4
    mov ebx, 1
    mov ecx, res
    mov edx, 1
    int 0x80

    mov eax, 1
    int 0x80



