"""Inizialize Variables."""


def variables():
    """Inizialize Variables."""
    dict_vars = dict()
    dict_vars['pathinp'] = 'C:\\_Python\\Data'
    dict_vars['pathout'] = 'C:\\_Python\\Report'

    fmtcategory = ['v22', 'v24', 'v30', 'v47', 'v52', 'v56',
                   'v66', 'v71', 'v74', 'v75', 'v79', 'v91',
                   'v107', 'v110', 'v112', 'v113', 'v125']

    fmtint = ['v62', 'v72', 'v129']

    fmtfloat = ['v1', 'v2', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9',
                'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17',
                'v18', 'v19', 'v20', 'v21', 'v23', 'v25', 'v26',
                'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35',
                'v36', 'v37', 'v39', 'v40', 'v42', 'v43', 'v44',
                'v45', 'v48', 'v49', 'v50', 'v51', 'v53', 'v54',
                'v55', 'v57', 'v58', 'v59', 'v60', 'v61', 'v63',
                'v65', 'v68', 'v69', 'v70', 'v73',
                'v76', 'v77', 'v78', 'v80', 'v81', 'v82', 'v83',
                'v84', 'v85', 'v86', 'v87', 'v88', 'v89', 'v90',
                'v93', 'v94', 'v95', 'v96', 'v97', 'v98', 'v99',
                'v100', 'v101', 'v102', 'v103', 'v104', 'v105',
                'v106', 'v108', 'v109', 'v111', 'v114', 'v115',
                'v116', 'v117', 'v118', 'v119', 'v120', 'v121',
                'v122', 'v123', 'v124', 'v126', 'v127', 'v128']

    dict_vars['key'] = ['ID']
    dict_vars['fmtcategory'] = fmtcategory[:]
    dict_vars['fmtint'] = fmtint[:]
    dict_vars['fmtfloat'] = fmtfloat[:]

    return dict_vars
