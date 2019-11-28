import textwrap

import numpy as np


def to_scalefont(fontsize=r"\normalsize"):
    """
    Scales the font in the tikzpicture.
    """
    text = rf"""
        \tikzset{{font={fontsize}}}
    """
    return textwrap.dedent(text=text)


def to_flatimage(pathfile, to="(-3,0,0)", width=8, height=8, name="temp"):
    """
    Adds a flat image to the canvas.
    """
    text = rf"""
        \node[canvas is xy plane at z=0] ({name}) at {to}{{
            \includegraphics[width={width}cm,height={height}cm]{{{pathfile}}}
        }};
    """
    return textwrap.dedent(text=text)


def to_curvedskip(of, to, angle=60, xoffset=0):
    """
    Curved skip connection arrow.
    Optionally, set the angle (default=60) for out-going and in-coming arrow,
    and an x offset for where the arrow starts and ends.
    """
    text = rf"""
        \draw [copyconnection]
            ([xshift={xoffset}cm] {of}-east)
            to [out={angle},in={180-angle}]
            node {{\copymidarrow}}
            ([xshift={xoffset}cm] {to}-east);
    """
    return textwrap.dedent(text=text)


def _Box(
    name,
    s_filer=256,
    n_filer=64,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2,
    height=40,
    depth=40,
    caption=" ",
    fill="ConvColor",
):
    """
    Generic Box used to draw Convolutional layers
    """
    xlabel = ", ".join(map(str, np.atleast_1d(n_filer)))
    width = ", ".join(map(str, np.atleast_1d(width)))
    text = rf"""
        \pic[shift={{ {offset} }}] at {to}{{
            Box={{
                name={name},
                caption={caption},
                xlabel={{ {xlabel}, }},
                zlabel={s_filer},
                fill={fill},
                height={height},
                width={{ {width} }},
                depth={depth}
                }}
            }};
        """
    return textwrap.dedent(text=text)


def _RightBandedBox(
    name,
    s_filer=256,
    n_filer=(64, 64, 64),
    offset="(0,0,0)",
    to="(0,0,0)",
    width=(2, 2, 2),
    height=40,
    depth=40,
    caption=" ",
    fill="ConvColor",
    bandfill="ConvReluColor",
):
    """
    Generic RightBandedBox used to draw Convolutional layers followed by something else
    """
    xlabel = ", ".join(map(str, np.atleast_1d(n_filer)))
    width = ", ".join(map(str, np.atleast_1d(width)))
    text = rf"""
        \pic[shift={{ {offset} }}] at {to}{{
        RightBandedBox={{
            name={name},
            caption={caption},
            xlabel={{ {xlabel}, }},
            zlabel={s_filer},
            fill={fill},
            bandfill={bandfill},
            height={height},
            width={{ {width} }},
            depth={depth}
            }}
        }};
    """
    return textwrap.dedent(text=text)


def to_InOut(**kwargs):
    """
    Input or Output Image Layer
    """
    return _Box(fill="{rgb:green,1;black,3}", **kwargs)


def to_RRDB(**kwargs):
    """
    Residual in Residual Dense Blocks
    """
    kwargs["n_filer"] = (" ",) * len(kwargs["n_filer"])  # remove x label
    return _Box(fill="{rgb:white,1;black,3}", **kwargs)


def to_ConvRelu(**kwargs):
    """
    Convolutional layer followed by a ReLU activation
    """
    return _RightBandedBox(fill=r"\ConvColor", bandfill=r"\ConvReluColor", **kwargs)


def to_Upsample(**kwargs):
    """
    Upsampling Layer (Nearest Neighbour).
    """
    return _Box(fill=r"\UnpoolColor", **kwargs)
