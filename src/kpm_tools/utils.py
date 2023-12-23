"""Helper functions.

This code is derived from the 'kwant' project, which is licensed under the
2-Clause BSD License. The original project can be found at:
https://kwant-project.org/

---
Copyright 2011-2015 C. W. Groth (CEA), M. Wimmer, A. R. Akhmerov,
X. Waintal (CEA), and others.  All rights reserved.
(CEA = Commissariat à l'énergie atomique et aux énergies alternatives)
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import inspect


def get_parameters(func):
    """Return the names of parameters of a function.

    It is made sure that the function can be called as `func(*args)` with
    `args` corresponding to the returned parameter names.

    Returns
    -------
    param_names : list
        Names of positional parameters that appear in the signature of 'func'.
    """

    def error(msg):
        fname = inspect.getsourcefile(func)
        try:
            line = inspect.getsourcelines(func)[1]
        except OSError:
            line = "<unknown line>"
        raise ValueError(f"{msg}:\nFile {repr(fname)}, line {line}, in {func.__name__}")

    p = inspect.Parameter
    pars = inspect.signature(func).parameters  # an *ordered mapping*
    names = []
    for k, v in pars.items():
        if v.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            if v.default is p.empty:
                names.append(k)
            else:
                error("Arguments of value functions " "must not have default values")
        elif v.kind is p.KEYWORD_ONLY:
            error("Keyword-only arguments are not allowed in value functions")
        elif v.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            error("Value functions must not take *args or **kwargs")
    return tuple(names)
