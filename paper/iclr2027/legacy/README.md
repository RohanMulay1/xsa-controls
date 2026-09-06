# Superseded two-column style

`iclr2026_conference_letx_twocolumn.sty` reproduced the geometry of an
unofficial two-column example (`format+Example.pdf` from letx.app). It is kept
because the measurement work in it is real and reusable, and because nothing
here is deleted.

It is not the venue's format. ICLR 2026 is single column, 5.5in wide, and the
paper now uses the official `iclr2026_conference.sty` from
<https://github.com/ICLR/Master-Template> unmodified.

## Superseded year

`iclr2026_conference.sty` and `.bst` are the 2026 files. The submission targets
ICLR 2027, whose style differs from 2026 only in the header year:

    88c88
    <     \lhead{Published as a conference paper at ICLR 2026}
    >     \lhead{Published as a conference paper at ICLR 2027}
    95c95
    <        \lhead{Under review as a conference paper at ICLR 2026}
    >        \lhead{Under review as a conference paper at ICLR 2027}

`fancyhdr.sty`, `natbib.sty`, `math_commands.tex` and the `.bst` are byte
identical across the two years. That is why the venue's compiled
`iclr2026_conference.pdf` is still a valid geometry reference for a 2027
submission: no dimension changed.
