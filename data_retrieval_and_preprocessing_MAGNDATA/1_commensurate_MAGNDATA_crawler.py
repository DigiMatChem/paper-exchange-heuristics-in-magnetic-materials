"""
Download commensurate MAGNDATA structures in mcif format.
"""

from bs4 import BeautifulSoup
import requests
import time


# Find commensurate MAGNDATA id's
magndata_ids = []
comm_url = ("https://www.cryst.ehu.es/magndata/index.php?adser=1&incomm_type_srch=comm&atoms_srch=&op_srch=AND"
            "&tot_species_srch=&auth_srch=&year_srch=&comments_srch=&msg_sys_srch=&parent_sys_srch=&msg_st_srch"
            "=&parent_st_srch=&0px=0px&1p0px=1p0px&1px=1px&2px=2px&3px=3px&1p1px=1p1px&transition_temp_srch=&"
            "experiment_temp_srch=&k_maximal_srch=&centrosym_srch=&polar_srch=&ferromagnetic_srch=&multiferro1_srch"
            "=&multiferro2_srch=&harmonic_srch=&same_point_srch=&tensor1=&tensor_logic1=AND&tensor2=&tensor_logic2="
            "AND&tensor3=&irrep_num_srch=&full_irrep_dim_srch=&small_irrep_dim_srch=&irrep_dim_srch=&direction_"
            "type_srch=&primary_srch=&secondary_srch=&presence_str_srch=&secondary_comm_srch=&submit=Search")

page = requests.get(comm_url, allow_redirects=True, timeout=20.00)
parsed_page = BeautifulSoup(page.text, "lxml")

for link in parsed_page.find_all("a"):
    if "?this_label=" in link.get("href"):
        magndata_ids.append(link.get("href").removeprefix("?this_label="))

for md_id in magndata_ids:
    # Find respective download link
    entry_url = "https://www.cryst.ehu.es/magndata/index.php?index=" + md_id
    page = requests.get(entry_url, allow_redirects=True, timeout=20.00)
    parsed_page = BeautifulSoup(page.text, "lxml")

    mcif_url = ["https://www.cryst.ehu.es/magndata/" + link.get("href") for link in parsed_page.find_all("a") if
                "mcif" in link.text]
    assert len(mcif_url) == 1, md_id + str(len(mcif_url))
    mcif_url = mcif_url[0]

    # Download mcif
    data = requests.get(mcif_url, allow_redirects=True, headers={"Referer": entry_url}, timeout=20.00)
    # if data.status_code == 200:
    with open(f"mcifs/{md_id}.mcif", "wb") as f:
        f.write(data.content)

    time.sleep(7)
