// PARecons.h : PARecons DLL ����ͷ�ļ�
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"		// ������


// CPAReconsApp
// �йش���ʵ�ֵ���Ϣ������� PARecons.cpp
//

class CPAReconsApp : public CWinApp
{
public:
	CPAReconsApp();

// ��д
public:
	virtual BOOL InitInstance();

	DECLARE_MESSAGE_MAP()
};
