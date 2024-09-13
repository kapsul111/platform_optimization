
%macro int_to_str 2
    mov esi, %1
    mov ecx, %2
    dec ecx

    call int_to_str_pr 
%endmacro

%macro write 2
    mov	eax, 4
    mov	ebx, 1
    mov	ecx, %1
    mov	edx, %2
    int	0x80
%endmacro

%macro write_newline 0
    write newline, newline_len
%endmacro 

section .data
    hello_msg  db  'Enter number', 0xa
    msg_len equ $ - hello_msg

    loop_msg  db  'Loop result ', 0xa
    loop_msg_len equ $ - loop_msg
    rec_msg  db  'Recursive result ', 0xa
    rec_msg_len equ $ - rec_msg

    newline db 0xa
    newline_len equ $ - newline

section .bss
    input resb 1
    res1 resb 16
    res2 resb 16

section .text
    global _start

_start:
    mov eax, 4
    mov ebx, 1
    mov ecx, hello_msg
    mov edx, msg_len
    int 0x80

    mov eax, 3
    mov ebx, 1
    mov ecx, input
    mov edx, 1
    int 0x80

    write loop_msg, loop_msg_len

    mov ecx, [input]
    sub ecx, '0'

    call loop_fact

    int_to_str res1, 16
    write res1, 16
    write_newline

    write rec_msg, rec_msg_len

    mov ecx, [input]
    sub ecx, '0'
    call recur_fact


    int_to_str res2, 16
    
    write res2, 16
    write_newline

EXIT:
    mov eax, 1
    int 0x80
    
loop_fact:
    mov eax, 1

reg_loop:
    mul ecx
    loop reg_loop

    ret

recur_fact:
    mov eax, 1
    call recur_fact_inside
    ret

recur_fact_inside:
    cmp ecx, 1
    je recur_end

    mul ecx
    dec ecx

    call recur_fact_inside

    recur_end:
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
    int_to_str_done:
    ret  