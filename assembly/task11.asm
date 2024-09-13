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

segment .data
    value_str db "-100+52-3"
    value_len equ $ - value_str

    stack dd 32 dup(0)
    stack_ptr dd stack
    base_stack_ptr dd stack
    newline db 0xa
    newline_len equ $ - newline

section .bss
    res resb 8

segment .text
global _start

_start:
    mov esi, value_str
    mov ecx, value_len
    add esi, ecx
    dec esi

    mov edi, 1

    parse_character:
        ;ebx - accumulator of all digits
        ;eax - current character, value of digit or operator in the end of loop
        ;edi - multiplicator of  second, third etc digit 
        ;esi - source str
        ;ecx - loop count
        mov eax, 0
        mov al, [esi]

        cmp al, '0'
        jl save_operation
        cmp al, '9'
        jg save_operation

    ;parse_digit
        sub al, '0'
        mul edi

        add ebx, eax
        
        mov eax, edi
        mov edi, 10
        mul edi

        mov edi, eax
        
        cmp esi, value_str
        je save_operand
        
        jmp next_iteration

    save_operation:
        cmp esi, value_str
        jne not_first_sign

        cmp al, '-'
        jne save_operand

        neg ebx
        push ebx

        jmp next_iteration

    not_first_sign:
        mov edx, eax
        call push_custom

    save_operand:
        push ebx
        mov ebx, 0
        mov edi, 1

    next_iteration:
        dec esi
    loop parse_character
    
calculate:
    mov edx, [stack_ptr]
    cmp edx, [base_stack_ptr]
    jle output

    call pop_custom

    cmp dl, '+'
    je plus
    
    cmp dl, '-'
    je minus

    mov edx, ebx
    call push_custom

plus:
    pop eax
    pop ebx
    add eax, ebx
    push eax

    jmp calculate

minus:
    pop eax
    pop ebx
    sub eax, ebx
    push eax

    jmp calculate

output:
    pop eax

    int_to_str res, 8
    write res, 8
    write_newline

done:
    mov eax, 1
    int 0x80

push_custom:
    push eax
    push ebx
    push ecx
    ;edx - value
    mov ebx, [stack_ptr]
    mov byte [ebx], dl
    inc ebx
    mov [stack_ptr], ebx

    pop ecx
    pop ebx
    pop eax

    ret
    
pop_custom:
    ; edx - return
    mov edx, 0
    mov ebx, [stack_ptr]
    dec ebx
    mov dl, [ebx]
    mov [stack_ptr], ebx

    ret

int_to_str_pr:
    push eax

    add esi, ecx
    mov ebx, 10

    cmp eax, 0
    jge int_to_str_loop
    neg eax

    int_to_str_loop:
        cmp eax, 0
        je check_sign
        mov edx, 0
        div ebx
        add dl, '0'
        mov byte [esi], dl
        dec esi
    loop int_to_str_loop

check_sign:
    pop eax
    cmp eax, 0
    jge int_to_str_done

    mov byte [esi], '-'

int_to_str_done:
    ret  