import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clem_dataclasses import AllSitesDataCollection, SiteDataSummary
from clem_general import ExecutiveControls


class DummyTEM:
    def __init__(self):
        self.output_root = os.getcwd()


class DummyMRCReader:
    pass


def test_executive_controls_registers_site_data_and_image_aliases():
    site_collection = AllSitesDataCollection()
    exc = ExecutiveControls(
        tem_communication=DummyTEM(),
        mrc_reader=DummyMRCReader(),
        sample_type="airyscan",
        milling_angle=0.0,
        site_collection=site_collection,
    )

    site_data = SiteDataSummary(site_id="site_001", path="/tmp/site_001")
    exc.register_site_data(site_data)

    assert site_collection.get_site("site_001") is site_data
    assert site_collection.active_site_id == "site_001"
    assert site_data.site_collection is site_collection

    image = np.arange(12).reshape(3, 4)
    site_data.image = image

    assert site_data.image is image
    assert site_data.mrc_full is image
    assert site_data.mrc.image is image
