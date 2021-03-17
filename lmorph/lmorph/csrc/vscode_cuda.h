#pragma once

// From https://gist.github.com/ruofeidu/df95ba27dfc6b77121b27fd4a6483426
#ifdef __INTELLISENSE__
void __syncthreads(); // workaround __syncthreads warning
#define KERNEL_ARG2(grid, block)
#define KERNEL_ARG3(grid, block, sh_mem)
#define KERNEL_ARG4(grid, block, sh_mem, stream)
#else
// clang-format off
#define KERNEL_ARG2(grid, block) <<<grid, block>>>
#define KERNEL_ARG3(grid, block, sh_mem) <<<grid, block, sh_mem>>>
#define KERNEL_ARG4(grid, block, sh_mem, stream) <<<grid, block, sh_mem, stream>>>
// clang-format on
#endif // __INTELLISENSE__
