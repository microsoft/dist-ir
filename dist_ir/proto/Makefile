all:
	protoc -I=. --python_out=. ./device.proto
	protoc -I=. --python_out=. ./type.proto
	protoc -I=. --python_out=. ./value.proto
	protoc -I=. --python_out=. ./op.proto
	protoc -I=. --python_out=. ./function.proto

clean:
	rm *_pb2.py
