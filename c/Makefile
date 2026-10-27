CC      ?= gcc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra
LDLIBS  = -lm

sensor: sensor.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

run: sensor
	./sensor ..

check: sensor
	./sensor --selftest ..

# AddressSanitizer + UndefinedBehaviorSanitizer build for memory/heap debugging
debug: sensor.c
	$(CC) -std=c99 -g -O0 -Wall -Wextra -fsanitize=address,undefined -o sensor_debug $< $(LDLIBS)
	./sensor_debug --selftest .. && ./sensor_debug ..

clean:
	rm -f sensor sensor_debug results.md

.PHONY: run check debug clean
