#pragma once

#define DUTIL_MAX_COL_NAME 64			// Maximum number of characters for the column name
#define DUTIL_MAX_DATA_BUFFER	32768	// Maximum data buffer that can be allocated. 32KB				
#define DUTIL_STRING_SEP_CHAR 2		// String separator character

enum enum_DUTIL_errors { 
	DUTIL_SUCCESS=0,
	DUTIL_BAD_ARGUMENTS = -1,
	DUTIL_NO_DATA = -2
};

typedef enum mDUTIL_Types {	
	DUTIL_ANY				= 0,
	DUTIL_CHAR			= 1,
	DUTIL_SHORT			= 2,
	DUTIL_LONG			= 4,
	DUTIL_FLOAT			= 7,
	DUTIL_DOUBLE			= 8,
	DUTIL_CHAR_NULL_LIST	= 11,
	DUTIL_CHAR_SEP_LIST	= 12
} DUTIL_Types;

char* newString(void* source, unsigned int type);
void* copyVariable(void* source, unsigned int type);
int compareVariables(void* var1, void* var2, unsigned int type);
void* strToVoidStar(char* source, unsigned int type);

class FixedList
{
public:
	unsigned int nCount; 
	DUTIL_Types type;

public:
	void** item;
	int		freeItems;

public:
	FixedList();
	FixedList(unsigned int maxSize);
	FixedList(unsigned int maxSize, DUTIL_Types dtype);
	~FixedList();	
	int AddItem(void* pData);							// Returns the list count
	unsigned int Count();								// Returns Count
	int QuickSort();
	void Empty();
	int FindFirst(void* valTofind);

private:
	void Swap(unsigned int i,unsigned  int j);
	void Qsort(unsigned int L, unsigned int R);									
};

typedef struct mListItem
{
	void* Data;				// Data content of the node. 
	struct mListItem* Next; // Pointer to the next node in the Doubly Linked List.							// NULL implies that this node is the last node in the list.
}ListItem;

/*
*******************************************************************************
* Class Name : LinkedList                                                     *
* Description: Implements the list. Maintains the list and implements the     *
*			   functions.                                                     *
*******************************************************************************
*/
class ListClass : public FixedList
{
private:
	ListItem* firstItem;	//Pointer to the first item in the list
	ListItem* lastItem;		//Pointer to the cuurent item in the list		

	ListItem* GetNthItem(unsigned int index);

public:
		
	// constructor: Initialize pointers to NULL
	ListClass();	
	~ListClass();
	int Add(void* pData);								// Returns the list count	
	int Insert(void* pData, unsigned int afterIndex);	// Returns the index where the new item is inserted.
	int Remove(unsigned int index);						// Returns Count
	int Remove(void* pData);							// Returns Count
	int RemoveAll(int freeMem);							// Returns 0
	void* GetNth(unsigned int index);					// Returns *pData
	void* SetNth(unsigned int index, void* pData);		// Returns *pData
	unsigned int Count();								// Returns Count		
	unsigned int Randomize();
};

class TableRow
{
public:
	unsigned int numColumns;
	void		 **fields;		// Array of a row fields. The length is = numColumns
	TableRow*	 nextRow;		// Points to the next row (in the linked list).

public:
	TableRow();
	TableRow(unsigned int numCols);		
	~TableRow();	
};

class TableColInfo
{	
public:
	char			name[DUTIL_MAX_COL_NAME];		// Name of the column
	unsigned int	maxSize;	// Maximum size of the data in the column. Its the size of data buffer
	DUTIL_Types		type;		// Data type
	unsigned int	decimals;	// Number of decimal places
	unsigned int	nullable;	// Nullable
	unsigned int	size;		// Current row data size. Its the size of the valid data in the data buffer. 
								// If data is string, then the size includes the null termination. dataSize of 0 
								// or less indicate null data.
	void			*data;		// Temporary Data buffer

	double			scale;
	double			shift;
	double			min;
	double			max;
	unsigned int	numDscrLvls;
	double*			dscrLvls;
	FixedList*		space;

public:									
	TableColInfo();	
	TableColInfo(char* colName, DUTIL_Types type, int maxSize);
	~TableColInfo();

	int setData(void* newData, int size);
	int getDataSize();	
};

class TableClass
{
public:
	unsigned int		numColumns;			// Number of Columns
	TableColInfo**		columnsInfo;		// Array of COLINF structure
	TableRow**			rows;
	unsigned int		numRows;			// Number of rows read SO FAR.
	int					freeColInfoOnDel;

private:
	TableRow*			firstRow;			// Points to the first row
	TableRow*			currentRow;			// Points to the last read row		


public:
	TableClass();		
	TableClass(TableClass* inputData, unsigned int splitAttrib, void* splitValue, char option);
	~TableClass();

	FixedList* GetAttribSpace(unsigned int attrib);
	FixedList* GetAttribSpaceCopy(unsigned int attrib);

	int GetAttribValCount(unsigned int attrib, void* val);	

	int getColIndex(char* attribName);
	int Normalize(int normOption);
	int ApplyDiscretization(unsigned int colIndex);
	int DiscretizeEqualWidth(unsigned int colIndex, unsigned int numIntervals);
	int DiscretizeEqualFreq(unsigned int colIndex, unsigned int numIntervals);
	int GetSpace();
//private:

	/******************************************************************************
	/ Function
	/ Name : AddRowFromColData
	/ Description : Adds a new row. The values come from values currently in 
	/				columnsInfo. 
	/*****************************************************************************/
	int AddRowFromColData();

	int AddRow(TableRow** ret);

	/******************************************************************************
	/ Function
	/ Name : GetFields
	/ Description : Gets an array of fields. 
	/*****************************************************************************/
	int GetFields(unsigned int row, void ***value);

	/******************************************************************************
	/ Function
	/ Name : GetField
	/ Description : Gets a field from a result set. type and size arguments can be null.
	/               The memory returned in value does not need to be deallocated.                
	/*****************************************************************************/
	int GetField(unsigned int row, unsigned int column, void **value, char **name, unsigned int *type, unsigned int *size);
	
	void FreeColumnsInfo();
	int MakeRowsArray();
	int MakeRowsArrayRandomized();
};

class ODBCTableClass : public TableClass
{

private:
	void *dbHandle;							// Database Handle
	void *connectionHandle;					// Connection Handle
	void *resultHandle;

public:
	ODBCTableClass();	
	ODBCTableClass(char* conStr);
	ODBCTableClass(char* conStr, char* sqlStr);	
	~ODBCTableClass();
	int Connect(char* conStr);	
	int Execute(char *sql);
	int StoreAll();
	int FetchRow();
	int FreeResultHandle();
	int Fetch();
	int Close();

private:
	DUTIL_Types  MapType(int type);		
};