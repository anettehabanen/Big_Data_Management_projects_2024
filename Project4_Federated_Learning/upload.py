from fedn import APIClient

host = "fedn.scaleoutsystems.com/bdm-ihy-fedn-reducer"
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzE5Mzg5NDkzLCJpYXQiOjE3MTY3OTc0OTMsImp0aSI6IjEwNDk3Yzg5MWFhMjQyNmQ5YjdkZDUwYjM3OGQ4MTRlIiwidXNlcl9pZCI6MTM4LCJjcmVhdG9yIjoiaGVsZW5hc29rayIsInJvbGUiOiJhZG1pbiIsInByb2plY3Rfc2x1ZyI6ImJkbS1paHkifQ.PiJnYR9i8YAoC_2Bd3Gtkqxv57xhVWvnQJRC5FHPkNI"

client = APIClient(host=host, token=token, secure=True, verify=True)
client.set_active_package("package.tgz", helper="numpyhelper")
client.set_active_model("seed.npz")
