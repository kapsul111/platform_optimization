%macro open_file 1
    mov eax, 5
    mov ebx, %1
    mov ecx, 2
    mov edx, 0777o  ; File permissions in octal
    int 0x80
%endmacro

%macro create_file 1
    mov  eax, 8
    mov  ebx, %1
    mov  ecx, 0777o  ; File permissions in octal
    int  0x80 
%endmacro

%macro read_file 3
    mov eax, 3
    mov ebx, %1 ;file descriptor
    mov ecx, %2 ;input buffer
    mov edx, %3 ;buffer size
    int 0x80
%endmacro

%macro write_file 3
    mov ebx, %1 ;file descriptor
    mov ecx, %2 ;output buffer
    mov edx, %3 ;buffer size
    mov eax, 4
    int 0x80
%endmacro

%macro close_file 1
    mov eax, 6
    mov ebx, %1 ;key
    int 0x80
%endmacro

%macro encript 3
    mov eax, %1 ;key
    mov ebx, %2 ;buffer
    mov ecx, %3 ;buffer size
    call encript_pr
%endmacro

%macro write 2
    mov	eax, 4
    mov	ebx, 1
    mov	ecx, %1
    mov	edx, %2
    int	0x80
%endmacro

%macro check_error 2
    mov ebx, %1
    mov ecx, %2
    call check_error_pr
%endmacro 

%macro read_input 2
    mov	eax, 3
    mov	ebx, 1
    mov	ecx, %1
    mov	edx, %2
    int	0x80
%endmacro

section .bss
    input_descriptor resd 1
    output_descriptor resd 1
    
    file_input resb 256
    file_output resb 256
    key resb 5

section .data
    filename_len dw 256
    buf db 8 dup(0)
    buf_size dd 1024
    buf_offset db 0

    source_msg db 'Enter source  filename', 0xa
    source_len equ $ - source_msg

    target_msg db 'Enter target filename', 0xa
    target_len equ $ - target_msg
    
    key_msg db 'Enter encryption key (4 letter long)', 0xa
    key_len equ $ - key_msg

    done db 'done', 0xa
    done_len equ $ - done
    error db 'error', 0xa
    error_len equ $ - error
    
section .text
    global _start

_start:
    write source_msg, source_len

    ;read source input
    read_input file_input, [filename_len]
    cmp eax, -1
    je error_handle
    
    ;replace /n to 0
    mov esi, file_input
    mov ecx, [filename_len]
    call replace_newline  

    write target_msg, target_len

    ;read target input
    read_input file_output, [filename_len]
    cmp eax, -1
    je error_handle
    
    ;replace /n to 0
    mov esi, file_output
    mov ecx, [filename_len]
    call replace_newline  

    write key_msg, key_len

    ;read key input
    read_input key, 4
    cmp eax, 4
    jne error_handle

    create_file file_output
    check_error error, error_len

    open_file file_input
    mov dword [input_descriptor], eax
    check_error error, error_len

    open_file file_output
    mov dword [output_descriptor], eax
    check_error error, error_len

read_write_loop:
    read_file [input_descriptor], buf, [buf_size]
    check_error error, error_len
    
    push eax
    ; encript
    mov ecx, [esp]
    mov ebx, buf
    mov eax, [key]
    call encript_pr

    ;write_file
    mov ebx, [output_descriptor]
    mov ecx, buf
    mov edx, [esp]
    mov eax, 4
    int 0x80
    
    check_error error, error_len

    pop eax

    cmp eax, [buf_size]
    je read_write_loop 

    write done, done_len

exit:
    close_file [input_descriptor]
    close_file [output_descriptor]

    mov eax, 1
    int 0x80

encript_pr:
    ;eax key
    ;ebx buffer
    ;ecx size

    cmp ecx, 4
    jle l1

l_big:
    mov edx, [ebx]
    xor edx, eax
    mov dword [ebx], edx
    add ebx, 4

    sub ecx, 4
    cmp ecx, 4
    jle l1

l1:
    mov dl, [ebx]
    xor dl, al
    mov byte [ebx], dl
    add ebx, 1

    loop l1 

    ret

check_error_pr:
    cmp eax, 0
    jge no_error

error_handle:
    write error, error_len
    
    jmp exit

no_error:
    ret

replace_newline:
    lodsb
    cmp al, 10  ; newline character
    je found_end
    loop replace_newline
found_end:
    dec esi
    mov byte [esi], 0  ; null terminator

    ret