import os
import tempfile

import numpy as np

from cvpipeline import image_filters
from cvpipeline import opticchiasm as oc
from cvpipeline import processsteps
from ezcomms import vnavs_data as vdata


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_test_filter(name, code, parms=None, flags=None, annotate_code=None):
    if parms is None:
        parms = []
    filt = image_filters.ImageFilter(name, code, parms, flags=flags)
    if annotate_code is not None:
        filt.annotate_code = annotate_code
    return filt


def _cleanup_filter(name):
    if name in image_filters.ImageFilterCollection.image_filters:
        del image_filters.ImageFilterCollection.image_filters[name]
    if name in image_filters.ImageFilterCollection.image_filter_names:
        image_filters.ImageFilterCollection.image_filter_names.remove(name)


# ---------------------------------------------------------------------------
# Tests: PipelineStep.configure_filter
# ---------------------------------------------------------------------------

def test_pipeline_step_configure_filter():
    step = processsteps.PipelineStep()
    step.configure_filter("Image")
    assert step.cv_filter_name == "Image"
    assert step.cv_specs is not None
    assert len(step.parms_specs) > 0


def test_pipeline_step_configure_filter_fills_defaults():
    step = processsteps.PipelineStep(filter_name="Image")
    # Image filter has max_width and max_height parms
    assert "Image_max_width" in step.parm_values
    assert "Image_max_height" in step.parm_values


def test_pipeline_step_configure_filter_show_annotation_injected():
    """Filters with annotate_code get ShowAnnotation appended if missing."""
    fname = "_test_ps_annot_inject"
    try:
        _make_test_filter(
            fname,
            "{x_output_im} = im_in.copy()",
            flags=[],
            annotate_code="{x_output_annotated} = im_in.copy()\n",
        )
        step = processsteps.PipelineStep(filter_name=fname)
        parm_names = [p.name for p in step.parms_specs]
        assert processsteps.SHOW_ANNOTATION in parm_names
    finally:
        _cleanup_filter(fname)


# ---------------------------------------------------------------------------
# Tests: PipelineStep.get_code_str
# ---------------------------------------------------------------------------

def test_pipeline_step_get_code_str_script_mode():
    fname = "_test_ps_script"
    try:
        _make_test_filter(
            fname,
            "{x_output_im} = xstep.source_im.copy()",
            flags=[image_filters.FLAG_ISBASE],
        )
        step = processsteps.PipelineStep(filter_name=fname)
        result = step.get_code_str(script=True)
        assert "im_in = xstep.source_im.copy()" in result
        assert "im_base = im_in" in result
    finally:
        _cleanup_filter(fname)


def test_pipeline_step_get_code_str_exec_mode():
    fname = "_test_ps_exec"
    try:
        _make_test_filter(
            fname,
            "{x_output_im} = xstep.source_im.copy()",
            flags=[image_filters.FLAG_ISBASE],
        )
        step = processsteps.PipelineStep(filter_name=fname)
        result = step.get_code_str(script=False)
        assert "xstep.exec_im = xstep.source_im.copy()" in result
        assert "im_base" not in result
    finally:
        _cleanup_filter(fname)


def test_pipeline_step_get_code_str_with_parm():
    fname = "_test_ps_parm"
    try:
        parm = vdata.DataAttribInt("threshold", "128")
        _make_test_filter(
            fname,
            "{x_output_im} = oc.threshold(im_in, {threshold})",
            parms=[parm],
            flags=[],
        )
        step = processsteps.PipelineStep(filter_name=fname)
        step.parm_values[fname + "_threshold"] = "42"
        result = step.get_code_str(script=True)
        assert "42" in result
    finally:
        _cleanup_filter(fname)


# ---------------------------------------------------------------------------
# Tests: PipelineStep.execute_step
# ---------------------------------------------------------------------------

def test_pipeline_step_execute_step():
    """Step 0 uses the real Image filter; exec_im gets populated from source_im."""
    source = oc.Image(im=np.zeros((10, 10, 3), dtype=np.uint8), colorcode="BGR")
    step = processsteps.PipelineStep(filter_name=image_filters.FILTER_NAME_IMAGE, ix=0)
    step.source_im = source
    trace, elapsed = step.execute_step([step])
    assert trace is None
    assert step.exec_im is not None
    assert step.exec_im.im.shape == (10, 10, 3)


def test_execute_pipeline_two_steps():
    fname1 = "_test_ps_pipe_s1"
    try:
        _make_test_filter(
            fname1,
            "{x_output_im} = im_in.copy()",
            flags=[],
        )
        source = oc.Image(im=np.zeros((8, 8, 3), dtype=np.uint8), colorcode="BGR")
        step0 = processsteps.PipelineStep(
            filter_name=image_filters.FILTER_NAME_IMAGE, ix=0
        )
        step0.source_im = source
        step1 = processsteps.PipelineStep(filter_name=fname1, ix=1)
        last, error = processsteps.execute_pipeline([step0, step1])
        assert error is None
        assert last is step1
        assert step1.exec_im is not None
        assert step1.exec_im.im.shape == (8, 8, 3)
    finally:
        _cleanup_filter(fname1)


def test_execute_pipeline_error_returns_trace():
    fname = "_test_ps_pipe_err"
    try:
        _make_test_filter(
            fname,
            "raise ValueError('boom')\n",
            flags=[],
        )
        source = oc.Image(
            im=np.zeros((4, 4, 3), dtype=np.uint8), colorcode="BGR"
        )
        step0 = processsteps.PipelineStep(
            filter_name=image_filters.FILTER_NAME_IMAGE, ix=0
        )
        step0.source_im = source
        step1 = processsteps.PipelineStep(filter_name=fname, ix=1)
        last, error = processsteps.execute_pipeline([step0, step1])
        assert error is not None
        assert "boom" in error
        assert last is step1
    finally:
        _cleanup_filter(fname)


# ---------------------------------------------------------------------------
# Tests: load_cvp_file / save_cvp_file round-trip
# ---------------------------------------------------------------------------

def test_load_cvp_file(tmp_path):
    cvp_file = tmp_path / "test.cvp"
    cvp_file.write_text("/Image\nparm.Image_max_width=640\nparm.Image_max_height=480\n")
    steps = processsteps.load_cvp_file(str(cvp_file))
    assert len(steps) == 1
    assert steps[0].cv_filter_name == "Image"
    assert steps[0].parm_values.get("Image_max_width") == "640"
    assert steps[0].parm_values.get("Image_max_height") == "480"


def test_load_cvp_file_multiple_steps(tmp_path):
    cvp_file = tmp_path / "multi.cvp"
    cvp_file.write_text("/Image\n/Gray\n")
    steps = processsteps.load_cvp_file(str(cvp_file))
    assert len(steps) == 2
    assert steps[0].cv_filter_name == "Image"
    assert steps[1].cv_filter_name == "Gray"
    assert steps[0].ix == 0
    assert steps[1].ix == 1


def test_save_cvp_file_round_trip(tmp_path):
    # Create steps
    step0 = processsteps.PipelineStep(filter_name="Image", ix=0)
    step0.parm_values["Image_max_width"] = "320"
    step1 = processsteps.PipelineStep(filter_name="Gray", ix=1)
    # Save
    cvp_file = str(tmp_path / "roundtrip.cvp")
    processsteps.save_cvp_file(cvp_file, [step0, step1])
    # Load
    loaded = processsteps.load_cvp_file(cvp_file)
    assert len(loaded) == 2
    assert loaded[0].cv_filter_name == "Image"
    assert loaded[1].cv_filter_name == "Gray"
    assert loaded[0].parm_values.get("Image_max_width") == "320"


def test_load_cvp_file_unknown_filter(tmp_path):
    cvp_file = tmp_path / "bad.cvp"
    cvp_file.write_text("/NoSuchFilter99\n")
    try:
        processsteps.load_cvp_file(str(cvp_file))
        assert False, "Expected KeyError"
    except KeyError:
        pass


# ---------------------------------------------------------------------------
# Tests: info_data methods
# ---------------------------------------------------------------------------

def test_info_data_methods():
    step = processsteps.PipelineStep()
    assert step.info_data == []
    ix0 = step.add_info("label_a", "value_a")
    assert ix0 == 0
    ix1 = step.add_info("label_b", "value_b")
    assert ix1 == 1
    assert step.info_data == [("label_a", "value_a"), ("label_b", "value_b")]
    step.set_info(0, "new_a", "new_va")
    assert step.info_data[0] == ("new_a", "new_va")
    step.set_info(4, "far", "away")
    assert len(step.info_data) == 5
    assert step.info_data[4] == ("far", "away")
    step.clear_info()
    assert step.info_data == []
