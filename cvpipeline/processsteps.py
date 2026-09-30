import codecs
import time
import traceback

import cv2
import numpy as np

from cvpipeline import image_filters
from cvpipeline import opticchiasm as oc
from ezcomms import vnavs_data as vdata

SHOW_ANNOTATION = "ShowAnnotation"


class PipelineStep:
    """Headless image-processing step — no Tkinter dependency.

    Holds filter state, parameter values, and execution outputs.
    A pipeline is just a plain ``list[PipelineStep]``.
    """

    __slots__ = (
        "cv_filter_name",
        "cv_specs",
        "exec_annotated",
        "exec_contours",
        "exec_hierarchy",
        "exec_hsvspec",
        "exec_im",
        "exec_objects",
        "exec_rect",
        "info_data",
        "ix",
        "parm_values",
        "parms_specs",
        "source_im",
        "source_path",
    )

    imports = [
        ("cv2", cv2, None),
        ("np", np, "numpy"),
        ("oc", oc, "oc"),
    ]

    def __init__(self, filter_name=None, parms=None, ix=0):
        self.ix = ix
        self.cv_filter_name = None
        self.cv_specs = None
        self.parm_values = parms if parms is not None else {}
        self.parms_specs = []
        self.exec_annotated = None
        self.exec_contours = None
        self.exec_hierarchy = None
        self.exec_hsvspec = None
        self.exec_im = None
        self.exec_objects = None
        self.exec_rect = None
        self.info_data = []
        self.source_im = None
        self.source_path = None
        if filter_name is not None:
            self.configure_filter(filter_name)

    def configure_filter(self, filter_name, new_parms=None):
        """Look up *filter_name* and populate parameter specs/defaults.

        This is the non-GUI core of ``ProcessStep.set_filter()``.
        """
        if new_parms is not None:
            for key, value in new_parms.items():
                self.parm_values[filter_name + "_" + key] = value
        self.cv_filter_name = filter_name
        self.cv_specs = image_filters.ImageFilterCollection.image_filters[filter_name]
        self.parms_specs = list(self.cv_specs.parms)  # copy so we can append
        if self.cv_specs.annotate_code is not None:
            annotation_control = False
            for this in self.parms_specs:
                if this.name == SHOW_ANNOTATION:
                    annotation_control = True
                    break
            if not annotation_control:
                self.parms_specs.append(
                    vdata.DataAttribBoolean(SHOW_ANNOTATION, "False")
                )
        # Fill defaults for any parms not already in parm_values
        for parm_spec in self.parms_specs:
            parm_name = self.cv_filter_name + "_" + parm_spec.name
            if parm_name not in self.parm_values:
                self.parm_values[parm_name] = parm_spec.default

    def get_code_str(self, script=True):
        """Generate executable code by substituting params into filter template.

        *script=True* uses ``im_in`` etc.; *script=False* uses ``xstep.exec_*``.
        """
        code_substitutions = {}
        show_annotation = False
        for this_parm in self.parms_specs:
            raw_value = self.parm_values[self.cv_filter_name + "_" + this_parm.name]
            translated_value = this_parm.GetValue(raw_value)
            code_substitutions[this_parm.name] = translated_value
            if this_parm.name == SHOW_ANNOTATION:
                show_annotation = translated_value
        if script:
            code_substitutions["x_output_annotated"] = "annotated"
            code_substitutions["x_output_contours"] = "contours_in"
            code_substitutions["x_output_hierarchy"] = "hierarchy_in"
            code_substitutions["x_output_hsvspec"] = "hsvspec_in"
            code_substitutions["x_output_im"] = "im_in"
            code_substitutions["x_output_objects"] = "objects_in"
            code_substitutions["x_output_rect"] = "rect_in"
        else:
            code_substitutions["x_output_annotated"] = "xstep.exec_annotated"
            code_substitutions["x_output_contours"] = "xstep.exec_contours"
            code_substitutions["x_output_hierarchy"] = "xstep.exec_hierarchy"
            code_substitutions["x_output_hsvspec"] = "xstep.exec_hsvspec"
            code_substitutions["x_output_im"] = "xstep.exec_im"
            code_substitutions["x_output_objects"] = "xstep.exec_objects"
            code_substitutions["x_output_rect"] = "xstep.exec_rect"
        code = self.cv_specs.code
        if code[-1:] != "\n":
            code += "\n"
        if script and (image_filters.FLAG_ISBASE in self.cv_specs.flags):
            code += "im_base = im_in\n"
        if self.cv_specs.annotate_code is not None:
            if show_annotation:
                code += "\n" + self.cv_specs.annotate_code
        if code != "":
            exec_code_str = code.format(**code_substitutions)
            return exec_code_str
        return ""

    def execute_step(self, steps):
        """Execute this step against *steps* (the full pipeline list).

        Returns ``(trace_or_None, elapsed_seconds)``.
        """
        execution_start = time.time()
        #
        # Collect output from prior steps
        #
        latest_base_image = None
        latest_im = None
        latest_contours = None
        latest_hierarchy = None
        latest_hsvspec = None
        latest_objects = None
        latest_rect = None
        #
        # Search prior steps to find the latest image,
        # contours, etc. that will be used as the inputs
        # for the current step. We don't know which inputs
        # the current step filter needs, so we just collect
        # whatever we find and have that available.
        #
        # At this time, exec_objects is a list of 
        # optichasm.LineObject() from hough_lines_p().
        # In the future it could be other things from other
        # filters (maybe).
        #
        # A nice enhancement would be to add the requirements to
        # the filter descriptions so we can verify what is needed
        # so we can print a nice specific message instead of
        # failing and listing a traceback.
        #
        for ix, this in enumerate(steps):
            if ix >= self.ix:
                break
            if this.exec_im is not None:
                latest_im = this.exec_im
                if image_filters.FLAG_ISBASE in this.cv_specs.flags:
                    latest_base_image = this.exec_im
            if this.exec_contours is not None:
                latest_contours = this.exec_contours
            if this.exec_hierarchy is not None:
                latest_hierarchy = this.exec_hierarchy
            if this.exec_hsvspec is not None:
                latest_hsvspec = this.exec_hsvspec
            if this.exec_objects is not None:
                latest_objects = this.exec_objects
            if this.exec_rect is not None:
                latest_rect = this.exec_rect

        #
        # Create environment for this step's execution
        #
        exec_global_vars = {}
        for this in self.imports:
            if this[1] is not None:
                exec_global_vars[this[0]] = this[1]
        exec_global_vars["xstep"] = self
        exec_global_vars["im_base"] = latest_base_image
        exec_global_vars["im_in"] = latest_im
        exec_global_vars["contours_in"] = latest_contours
        exec_global_vars["hierarchy_in"] = latest_hierarchy
        exec_global_vars["hsvspec_in"] = latest_hsvspec
        exec_global_vars["objects_in"] = latest_objects
        exec_global_vars["rect_in"] = latest_rect

        trace = None
        self.exec_annotated = None
        self.exec_contours = None
        self.exec_hierarchy = None
        self.exec_hsvspec = None
        self.exec_im = None
        self.exec_objects = None
        self.exec_rect = None
        no_image = (
            (self.cv_filter_name == image_filters.FILTER_NAME_IMAGE
             and self.source_im is None)
            or (self.cv_filter_name != image_filters.FILTER_NAME_IMAGE
                and latest_im is None)
        )
        if no_image:
            elapsed = time.time() - execution_start
            return ("No image found.", elapsed)
        exec_code_str = self.get_code_str(script=False)
        if exec_code_str != "":
            try:
                exec(exec_code_str, exec_global_vars)
            except:
                trace = traceback.format_exc()
        elapsed = time.time() - execution_start
        return (trace, elapsed)

    def clear_info(self):
        self.info_data = []

    def add_info(self, label, value):
        ix = len(self.info_data)
        self.info_data.append((label, value))
        return ix

    def set_info(self, ix, label, value):
        while len(self.info_data) < (ix + 1):
            self.info_data.append(("", ""))
        self.info_data[ix] = (label, value)


# ---------------------------------------------------------------------------
# Module-level functions
# ---------------------------------------------------------------------------

def load_cvp_file(fn):
    """Parse a ``.cvp`` file and return ``list[PipelineStep]``."""
    steps = []
    current_filter = None
    current_parms = {}

    def _assign():
        nonlocal current_filter, current_parms
        step = PipelineStep(
            filter_name=current_filter,
            parms=dict(current_parms),
            ix=len(steps),
        )
        steps.append(step)
        current_filter = None
        current_parms = {}

    with open(fn, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln == "":
                continue
            if ln[0] == "/":
                if current_filter is not None:
                    _assign()
                current_filter = ln[1:]
            else:
                sep = ln.find("=")
                if sep > 0:
                    key = ln[:sep][5:]  # eliminate "parm." prefix
                    value = ln[sep + 1:]
                    current_parms[key] = value
    if current_filter is not None:
        _assign()
    return steps


def execute_pipeline(steps, source_im=None, source_path=None):
    """Execute all *steps* in order.

    Returns ``(last_step, error_or_None)``.
    """
    if steps and source_im is not None:
        steps[0].source_im = source_im
        steps[0].source_path = source_path
    for step in steps:
        trace, elapsed = step.execute_step(steps)
        if trace is not None:
            return (step, trace)
    last = steps[-1] if steps else None
    return (last, None)


def save_cvp_file(fn, steps):
    """Write a ``.cvp`` file from *steps*."""
    with open(fn, "w") as f:
        for step in steps:
            f.write(f"/{step.cv_filter_name}\n")
            for key, value in step.parm_values.items():
                f.write(f"parm.{key}={value}\n")


def write_cameraman_script(steps):
    """Return a cameraman script string from *steps*."""
    lines = []
    lines.append("im_in = im_base.copy()\n")
    for step in steps[1:]:
        code_str = step.get_code_str(script=True)
        lines.append(code_str)
    if steps[-1].cv_specs.annotate_code is None:
        lines.append("display_image = im_in\n")
    else:
        lines.append("display_image = annotated\n")
    return "".join(lines)


def write_cameraman_file(cam_fn, steps):
    """Write a cameraman script file from *steps*."""
    script = write_cameraman_script(steps)
    with codecs.open(cam_fn, "w", encoding="utf-8") as f:
        f.write(script)
