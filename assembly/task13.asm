%macro addition 3
    mov eax, %1
    mov ebx, %2
    add eax, ebx
    mov dword [%3], eax
%endmacro

%macro subdevide 3
    mov eax, %1
    mov ebx, %2
    sub eax, ebx
    mov dword [%3], eax
%endmacro

%macro multiply 3
    mov ax, %1
    mov bx, %2
    mul bx
    shl edx, 16
    add eax, edx
    mov dword [%3], eax
%endmacro

%macro division 3
    mov edx, 0
    mov eax, %1
    mov ebx, %2
    div ebx
    mov dword [%3], eax
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

%macro int_to_str 2
    mov esi, %1
    mov ecx, %2
    dec ecx

    call int_to_str_pr 
%endmacro
section .data
    newline db 0xa
    newline_len equ $ - newline

section .bss
    add_res resd 1
    sub_res resd 1
    div_res resd 1
    mul_res resd 1
    res resb 16

section .text
    global _start

_start:
    addition 5, 3, add_res

    mov eax, [add_res]
    int_to_str res, 16
    write res, 16
    write_newline
    
    subdevide 5, 3, sub_res
    
    mov eax, [sub_res]
    int_to_str res, 16
    write res, 16
    write_newline

    mov eax, [mul_res]
    multiply 1000, 100, mul_res
    
    int_to_str res, 16
    write res, 16
    write_newline

    division 15, 5, div_res
    
    mov eax, [div_res]
    int_to_str res, 16
    write res, 16
    write_newline

exit:
    mov eax, 1
    xor ebx, ebx
    int 0x80

    
int_to_str_pr:
    add esi, ecx

    mov ebx, 10
    int_to_str_loop:
        cmp eax, 0
        je fill_zero
        mov edx, 0
        div ebx
        add dl, '0'
        mov byte [esi], dl
        dec esi
    loop int_to_str_loop
    fill_zero:
        mov byte [esi], 0
        dec esi
    loop fill_zero
    ret 