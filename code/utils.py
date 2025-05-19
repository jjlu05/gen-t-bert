import pickle
import pickle5 as p
import numpy as np

def loadDictionaryFromPickleFile(dictPath):
    ''' Load the pickle file as a dictionary
    Args:
        dictPath: path to the pickle file
    Return: dictionary from the pickle file
    '''
    filePointer=open(dictPath, 'rb')
    dictionary = p.load(filePointer)
    filePointer.close()
    return dictionary

def saveDictionaryAsPickleFile(dictToSave, dictPath):
    ''' Save dictionary as a pickle file
    Args:
        dictToSave to be saved
        dictPath: filepath to which the dictionary will be saved
    '''
    filePointer=open(dictPath, 'wb')
    pickle.dump(dictToSave,filePointer, protocol=pickle.HIGHEST_PROTOCOL)
    filePointer.close()
    
def loadListFromTxtFile(listPath): 
    ''' Save list as a text file
    Args:
        listPath: filepath with the stored list
    '''
    savedList = []
    with open(listPath) as f:
        listItems = f.read().splitlines()
        for i in listItems:
            savedList.append(i)
    return savedList
    
def saveListAsTxtFile(listToSave, listPath): 
    ''' Save list as a text file
    Args:
        listToSave to be saved
        listPath: filepath to which the list will be saved
    '''
    with open(listPath, "w") as output:
        for item in listToSave:
            output.writelines(str(item)+'\n')
            
def projectAtts(tableDfs, queryTable):
    '''For each table, project out the attributes that are in the query table.
       If the table has no shared attributes with the query table, remove it.
    '''
    projectedDfs = {}
    queryCols = queryTable.columns
    for table, df in tableDfs.items():
        tableCols = df.columns
        commonCols = [c for c in tableCols if c in queryCols]
        print(f"[projectAtts] {table}: common columns with query = {commonCols}")
        if commonCols:
            projectedDfs[table] = df[commonCols]
        else:
            print(f"[projectAtts] {table} removed due to no shared columns.")
    return projectedDfs
def selectKeys(tableDfs, queryTable, primaryKey, foreignKeys):
    '''For each table, select tuples that contain key value from queryTable.
       If the table has no shared keys with the queryTable, remove it.
    '''
    selectedDfs = {}
    queryKeyVals = {key: queryTable[key].tolist() for key in queryTable.columns}

    for table, df in tableDfs.items():
        dfCols = df.columns.tolist()
        commonKeys = [k for k in [primaryKey] + foreignKeys if k in dfCols]
        print(f"[selectKeys] {table}: candidate key columns = {commonKeys}")

        if not commonKeys:
            print(f"[selectKeys] {table} skipped — no matching keys.")
            continue

        # If only one key is found, we pick a second key with most non-null values
        if len(commonKeys) == 1:
            primary = commonKeys[0]
            bestSecondKey = None
            maxNonNulls = 0
            for col in dfCols:
                if col != primary and df[col].count() > maxNonNulls:
                    bestSecondKey = col
                    maxNonNulls = df[col].count()
            if bestSecondKey:
                commonKeys = [primary, bestSecondKey]
                print(f"[selectKeys] {table}: added best secondary key = {bestSecondKey}")

        # Debug overlap values
        for key in commonKeys:
            source_vals = set(str(val).strip().lower() for val in queryKeyVals.get(key, []))
            candidate_vals = set(str(val).strip().lower() for val in df[key])
            overlap = source_vals.intersection(candidate_vals)
            print(f"[DEBUG] {table}: Overlap on '{key}': {len(overlap)} values")

        # Apply filtering condition
        conditions = [df[commonKeys[0]].isin(queryKeyVals.get(commonKeys[0], [])).values]
        for key in commonKeys[1:]:
            conditions.append(df[key].isin(queryKeyVals.get(key, [])).values)

        mask = np.bitwise_or.reduce(conditions)
        selectedTuplesDf = df.loc[mask].drop_duplicates().reset_index(drop=True)

        if not selectedTuplesDf.empty:
            print(f"[selectKeys] {table} kept ({selectedTuplesDf.shape[0]} rows)")
            selectedDfs[table] = selectedTuplesDf
        else:
            print(f"[selectKeys] {table} removed — no matching rows after key filter.")
            
    return selectedDfs
