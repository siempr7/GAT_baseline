kg_info_prompt = """Let's start with some concepts of the urban knowledge graph, which captures facts related to a city. In the following, we will introduce the entities and relations in it.
The urban knowledge graph contains 7 types of entities, i.e., Brand, Point of Interest (POI), Region, Business Area, Category1(coarse-grained category), Category2(medium-grained category), Cateory3(fine-grained category). In this context, fine-grained categorys is subcategory of medium-grained category and medium-grained category is subcategory of coarse-grained category. The meaning of a specific entity is as follows.
The urban knowledge graph contains 35 types of relations to describe the connections between entities. The name of each relation consists of the entity type corresponding to the origin of this relation, the meaning of the relation, and the entity type corresponding to the destination of this relation, in order from left to right, separated by _, which indicates the relation to be experienced from the starting entity to the end entity. Below I will map the names of relations to IDs.
Mapping of relation names to IDs (The left side of the colon is the IDs and the right side is the names):
rel_1: POI_HasCategoryOf_Category1
rel_2: POI_HasCategoryOf_Category2
rel_3: POI_HasCategoryOf_Category3
rel_4: Category1_ExistIn_POI
rel_5: Category2_ExistIn_POI
rel_6: Category3_ExistIn_POI
rel_7: Brand_BelongTo_Category1
rel_8: Brand_BelongTo_Category2
rel_9: Brand_BelongTo_Category3
rel_10: Category1_HasBrandOf_Brand
rel_11: Category2_HasBrandOf_Brand
rel_12: Category3_HasBrandOf_Brand
rel_13: Region_Has_POI
rel_14: POI_LocateAt_Region
rel_15: POI_BelongTo_BusinessArea
rel_16: BusinessArea_Contain_POI
rel_17: Category2_IsSubCategoryOf_Category1
rel_18: Category3_IsSubCategoryOf_Category1
rel_19: Category3_IsSubCategoryOf_Category2
rel_20: Category1_IsBroadCategoryOf_Category2
rel_21: Category1_IsBroadCategoryOf_Category3
rel_22: Category2_IsBroadCategoryOf_Category3
rel_23: Region_PopulationInflowFrom_Region
rel_24: Region_PopulationFlowTo_Region
rel_25: Brand_HasPlacedStoreAt_Region
rel_26: Region_HasStoreOf_Brand
rel_27: Brand_ExistIn_POI
rel_28: POI_HasBrandOf_Brand
rel_29: Region_ServedBy_BusinessArea
rel_30: BusinessArea_Serve_Region
rel_31: Brand_RelatedBrand_Brand
rel_32: POI_Competitive_POI
rel_33: Region_BorderBy_Region
rel_34: Region_NearBy_Region
rel_35: Region_SimilarFunction_Region
"""

rel2name = {'rel_1': ['POI', 'Has Category Of', 'Category1'],
 'rel_2': ['POI', 'Has Category Of', 'Category2'],
 'rel_3': ['POI', 'Has Category Of', 'Category3'],
 'rel_4': ['Category1', 'Exists In', 'POI'],
 'rel_5': ['Category2', 'Exists In', 'POI'],
 'rel_6': ['Category3', 'Exists In', 'POI'],
 'rel_7': ['Brand', 'Belongs To', 'Category1'],
 'rel_8': ['Brand', 'Belongs To', 'Category2'],
 'rel_9': ['Brand', 'Belongs To', 'Category3'],
 'rel_10': ['Category1', 'Has Brand Of', 'Brand'],
 'rel_11': ['Category2', 'Has Brand Of', 'Brand'],
 'rel_12': ['Category3', 'Has Brand Of', 'Brand'],
 'rel_13': ['Region', 'Has', 'POI'],
 'rel_14': ['POI', 'Locates At', 'Region'],
 'rel_15': ['POI', 'Belongs To', 'BusinessArea'],
 'rel_16': ['BusinessArea', 'Contains', 'POI'],
 'rel_17': ['Category2', 'Is Sub-Category Of', 'Category1'],
 'rel_18': ['Category3', 'Is Sub-Category Of', 'Category1'],
 'rel_19': ['Category3', 'Is Sub-Category Of', 'Category2'],
 'rel_20': ['Category1', 'Is Broad Category Of', 'Category2'],
 'rel_21': ['Category1', 'Is Broad Category Of', 'Category3'],
 'rel_22': ['Category2', 'Is Broad Category Of', 'Category3'],
 'rel_23': ['Region', 'Has Large Population Inflow From', 'Region'],
 'rel_24': ['Region', 'Has Large Population Flow To', 'Region'],
 'rel_25': ['Brand', 'Has Placed Store At', 'Region'],
 'rel_26': ['Region', 'Has Store Of', 'Brand'],
 'rel_27': ['Brand', 'Exists In', 'POI'],
 'rel_28': ['POI', 'Has Brand Of', 'Brand'],
 'rel_29': ['Region', 'Is Served By', 'BusinessArea'],
 'rel_30': ['BusinessArea', 'Serves', 'Region'],
 'rel_31': ['Brand', 'Has Related Brand Of', 'Brand'],
 'rel_32': ['POI', 'Is Competitive With', 'POI'],
 'rel_33': ['Region', 'Is Border By', 'Region'],
 'rel_34': ['Region', 'Is Near By', 'Region'],
 'rel_35': ['Region', 'Has Similar Function With', 'Region']}

impact_aspects_prompt = kg_info_prompt + """Now we are using this knowledge graph for the "Population_Prediction" task, i.e., predicting the population in urban regions. Please help me find some relation paths in this KG which may help this task. (The rationality of the relation path must be ensured. For two neighboring relations in a relation path, the destination entity type of the former relation must be the same as the origination entity type of the latter relation)
Multiple Relation Paths: {top_paths}

{reference}
Your answer:  """

impact_aspects_prompt_jl = kg_info_prompt + """Now we are using this knowledge graph for the "Population_Prediction" task, i.e., predicting the population in urban regions. Please help me find some relation paths in this KG which may help this task. (The rationality of the relation path must be ensured. For two neighboring relations in a relation path, the destination entity type of the former relation must be the same as the origination entity type of the latter relation)
Multiple Relation Paths: {top_paths}

{reference}
Your Answer: """

impact_aspects_prompt_cl = kg_info_prompt + """Now we are using this knowledge graph for the "Population_Prediction" task, i.e., predicting the population in urban regions. Please help me find some relation paths in this KG which may help this task. (The rationality of the relation path must be ensured. For two neighboring relations in a relation path, the destination entity type of the former relation must be the same as the origination entity type of the latter relation)

The reference results:
{reference}

We provide the 10 optimal paths we have discovered for your reference. Please note that you must organize the content of the output from 5 aspects and only 1 relation path per aspect. The format (not the content) of the output must be the same as the reference above, in particular, "Relevant relation paths include: {{.....}}" must be outputted in every aspects:"""

