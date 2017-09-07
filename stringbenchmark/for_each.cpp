/***************************************************************************
 *   Copyright (C) 2007 by Guy Rutenberg                                   *
 *   guyrutenberg@gmail.com                                                *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <iostream>
#include "hugestring.h"
#include <algorithm>
using namespace std;

int main()
{
	int i=0;
	int num[26];

	for (i=0;i<26;i++)
		num[i]=0;

	for_each( cbegin(hugestring), cend(hugestring), [&num] (char letter)
	// note: arrays are copied element-wise in capture-by-value lambdas.
	// The array does not decay into a pointer, like with a function argument.
	{
		num[letter - 'a']++;
	});

	for (i=0; i<26; i++)
		cout<<(char)('a'+i)<<": "<<num[i]<<endl;
	return 0;
}
