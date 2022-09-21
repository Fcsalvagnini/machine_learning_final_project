help:
	cat Makefile
run:
	docker-compose up
exec:
	docker-compose run mo444 bash
build:
	docker-compose build
rebuild: stop .FORCE
	docker-compose build --force-rm
stop:
	docker stop mo444 || true; docker rm mo444 || true;
.FORCE: