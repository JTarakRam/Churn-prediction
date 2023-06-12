import os
from pathlib import Path
import streamlit as st
from typing import Dict
from typing import List
from typing import Text

from source.ui import display_header
from source.ui import display_report
from source.ui import select_report
from source.ui import set_page_container_style
from source.utils import EntityNotFoundError
from source.utils import get_reports_mapping

REPORTS_DIR_NAME: Text = "reports"

def main():
    st.set_page_config(layout='wide', page_title='Monitoring Reports')

    set_page_container_style()

    try:
        reports_dir: Path = Path(REPORTS_DIR_NAME)

        report_mapping: Dict[Text, Path] = get_reports_mapping(reports_dir)
        selected_report_name: Text = select_report(report_mapping)
        selected_report: Path = report_mapping[selected_report_name]

        display_header(selected_report_name)
        display_report(selected_report)

    except EntityNotFoundError as e:
        st.error(e)

    except Exception as e:
        raise e

if __name__ == "__main__":
    main()
