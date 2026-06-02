CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -Iinclude
LDFLAGS = -lm

SRC = src/tda.c
OBJ = $(SRC:.c=.o)

.PHONY: all test clean

all: libtda.a test_tda

libtda.a: $(OBJ)
	ar rcs $@ $^

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

test_tda: tests/test_tda.c libtda.a
	$(CC) $(CFLAGS) $< -L. -ltda $(LDFLAGS) -o $@

test: test_tda
	./test_tda

clean:
	rm -f src/*.o libtda.a test_tda
