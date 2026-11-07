# Calamine CFB regression inputs

The two malformed workbooks come from the upstream reproducer for
[RUSTSEC-2026-0436](https://rustsec.org/advisories/RUSTSEC-2026-0436.html),
[Calamine PR #762](https://github.com/tafia/calamine/pull/762), commit
`908d121414b6379dba07bad580d9674520ece9df`.
They exercise a truncated directory entry and a cyclic DIFAT chain through
Hayashi's public Excel loader. They are deliberately invalid test inputs.
The valid BIFF5 workbook from Calamine 0.38.0 is a control for loading ordinary
compound files, including their row count, headers and text values.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| [cfb_dir_short_chunk_panic.xls](https://raw.githubusercontent.com/tafia/calamine/908d121414b6379dba07bad580d9674520ece9df/tests/cfb_dir_short_chunk_panic.xls) | 4153 | `82de55b1a1c0ee69e8460e29172af5711a109f7ad04a75f7d112e938fe3ec9c7` |
| [cfb_difat_oom.xls](https://raw.githubusercontent.com/tafia/calamine/908d121414b6379dba07bad580d9674520ece9df/tests/cfb_difat_oom.xls) | 512 | `b0ff76967dfbb4e4870c848a6365c74f361e48bfd98afed1a5ed378b0ab2a6a8` |
| [biff5_write.xls](https://raw.githubusercontent.com/tafia/calamine/a4dba3b5309d81b2774018a13be39e68d938c3c7/tests/biff5_write.xls) | 10752 | `1807130ee9d96c1a738772e53445f9c9efc193042a994549ec6d092e5386a5f9` |

The upstream MIT notice is reproduced below for these copied fixtures.

The MIT License (MIT)

Copyright (c) 2016 Johann Tuffe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:


The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
