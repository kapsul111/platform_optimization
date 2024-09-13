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

%macro read 2
    mov	eax, 3
    mov	ebx, 1
    mov	ecx, %1
    mov	edx, %2
    int	0x80
%endmacro

%macro str_to_int 2
    xor eax, eax
    mov esi, %1
    mov ecx, %2
    dec ecx

    call str_to_int_pr
%endmacro

%macro int_to_str 2
    mov esi, %1
    mov ecx, %2
    dec ecx

    call int_to_str_pr 
%endmacro

section .data
    hello_msg db "Please, enter first number, then second number (both from 0 to 1000), than symbol of operation (+,-,*,/)", 0xa
    hello_len equ $ - hello_msg

    offer_msg db "Do you want to make another calculation? enter y or n", 0xa
    offer_len equ $ - offer_msg

    input_error_msg db "Wrong input - too big number (more than 1000) or nonnumerical data, wrong operation etc", 0xa
    input_error_msg_len equ $ - input_error_msg

    zero_error_msg db "Error: divizion by zero", 0xa
    zero_error_len equ $ - zero_error_msg

    newline db 0xa
    newline_len equ $ - newline
    
    f_msg db "first", 0xa
    f_len equ $ - f_msg
    s_msg db "second", 0xa
    s_len equ $ - s_msg
    o_msg db "op", 0xa
    o_len equ $ - o_msg
    res db 8 dup(0)

section .bss
    first resb 8
    second resb 8
    op resb 2
    offer_input resb 2

section .text
    global _start

_start:
    write hello_msg, hello_len

    read first, 8
    cmp eax, -1
    je input_error

    read second, 8
    cmp eax, -1
    je input_error

    read op, 2
    cmp eax, -1
    je input_error

    str_to_int second, 8
    push eax
    str_to_int first, 8
    pop ebx

    cmp eax, 1000
    jg input_error

    cmp ebx, 1000
    jg input_error

    mov cl, [op]
    cmp cl, '+'
    je sum

    cmp cl, '-'
    je subdevide

    cmp cl, '*'
    je multiply

    cmp cl, '/'
    je division

    jmp input_error

sum:
    add eax, ebx
    int_to_str res, 8
    jmp return_answer

subdevide:
    sub eax, ebx
    int_to_str res, 8
    jmp return_answer

multiply:
    mul ebx
    int_to_str res, 8
    jmp return_answer

division:
    mov edx, 0
    
    cmp ebx, 0
    je zero_error
    div ebx

    int_to_str res, 8
    jmp return_answer

return_answer:
    write res, 8
    write_newline

offering:
    write offer_msg, offer_len
    read offer_input, 2
    cmp byte [offer_input], 'y'
    je _start
    cmp byte [offer_input], 'n'
    je exit

    jmp offering

exit:
    mov eax, 1
    xor ebx, ebx
    int 0x80

input_error:
    write input_error_msg, input_error_msg_len
    jmp offering

zero_error:
    write zero_error_msg, zero_error_len
    jmp offering

str_to_int_pr:
    mov eax, 0
    mov ebx, 0
    str_to_int_loop:
        mov edx, 10
        mov bl, [esi]
        cmp bl, 0xa
        je str_to_int_done
        mul edx
        sub bl, '0'
        add eax, ebx

        inc esi
    loop str_to_int_loop
    str_to_int_done:
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