# Deep Euler Method Tests
Codes for testing the Deep Euler Method. The Deep Euler Method is a numerical integration scheme
which takes advantage of a neural network to approximate the local truncation error of the Euler Method.
By adding the approximation to the Euler-step a higher order solution is achieved.

Read about the Deep Euler Method in the paper:
[Deep Euler method: solving ODEs by approximating the local truncation error of the Euler method](https://arxiv.org/abs/2003.09573).

The same scheme was proposed in the paper [Hypersolvers: Toward Fast Continuous-Depth Models](https://papers.nips.cc/paper/2020/hash/f1686b4badcf28d33ed632036c7ab0b8-Abstract.html)

## Build instructions on windows
* Open Developer Command Prompt for VS2019
* Specify torch dir to cmake `Torch_DIR=xxx` (containing `TorchConfig.cmake`), provided by miniconda.
* `msbuild DEM.sln -p:Configuration=Release`

For example:

```cmd
cmake -DCMAKE_TOOLCHAIN_FILE=C:/Users/z50019165/Documents/repos/vcpkg/scripts/buildsystems/vcpkg.cmake -DTorch_DIR=C:\Users\z50019165\Miniconda3\envs\spinningup\Lib\site-packages\torch\share\cmake\Torch\ ..
```