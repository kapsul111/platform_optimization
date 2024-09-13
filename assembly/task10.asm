section .data
    data_str db "data2134", 0xa
    data_len equ $ - data_str

section .bss
    dest resb 16
    
section .text
    global _start

_start:
    mov esi, data_str
    mov edi, dest
    mov ecx, data_len
    call encode

    mov esi, dest
    mov ecx, data_len
    push eax
    call decode

    mov eax, 4
    mov ebx, 1
    mov ecx, dest
    mov edx, data_len

    xor ebx, ebx
    int 0x80

    mov eax, 1
    int 0x80

encode:
    encode_loop:
        mov bl, [esi]
        rol bl, 2
        mov byte [edi], bl
        inc esi
        inc edi
    loop encode_loop

    ret

decode:
    decode_loop:
        mov bl, [esi]
        ror bl, 2
        mov byte [esi], bl
        inc esi
    loop decode_loop

    ret

