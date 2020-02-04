
**Немного магии**

Magic commands -- функции, позволяющие решать некоторые стандартные проблемы


```python
%lsmagic
```




    Available line magics:
    %alias  %alias_magic  %autocall  %automagic  %autosave  %bookmark  %cd  %clear  %cls  %colors  %config  %connect_info  %copy  %ddir  %debug  %dhist  %dirs  %doctest_mode  %echo  %ed  %edit  %env  %gui  %hist  %history  %killbgscripts  %ldir  %less  %load  %load_ext  %loadpy  %logoff  %logon  %logstart  %logstate  %logstop  %ls  %lsmagic  %macro  %magic  %matplotlib  %mkdir  %more  %notebook  %page  %pastebin  %pdb  %pdef  %pdoc  %pfile  %pinfo  %pinfo2  %popd  %pprint  %precision  %profile  %prun  %psearch  %psource  %pushd  %pwd  %pycat  %pylab  %qtconsole  %quickref  %recall  %rehashx  %reload_ext  %ren  %rep  %rerun  %reset  %reset_selective  %rmdir  %run  %save  %sc  %set_env  %store  %sx  %system  %tb  %time  %timeit  %unalias  %unload_ext  %who  %who_ls  %whos  %xdel  %xmode
    
    Available cell magics:
    %%!  %%HTML  %%SVG  %%bash  %%capture  %%cmd  %%debug  %%file  %%html  %%javascript  %%js  %%latex  %%markdown  %%perl  %%prun  %%pypy  %%python  %%python2  %%python3  %%ruby  %%script  %%sh  %%svg  %%sx  %%system  %%time  %%timeit  %%writefile
    
    Automagic is ON, % prefix IS NOT needed for line magics.



line magics -- один ```%``` впереди и относятся к одной строке,

cell magics -- два ```%%``` впереди и относятся к ячейке.

```%matplotlib inline``` отображает графики в Jupyter notebook

```%matplotlib notebook``` добавляет интерактивность


```python
%matplotlib inline
import matplotlib.pyplot as plt
plt.plot([[0,0],[1,1]], linewidth = 2)
plt.show()
```


![png](output_4_0.png)



```python
%matplotlib notebook
import matplotlib.pyplot as plt
plt.plot([[0,0],[1,1]], linewidth = 2)
plt.show()
```


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAbAAAAEgCAYAAADVKCZpAAAgAElEQVR4nO3de3SU9b3v8Y/V3e6119qDiNeqKWg9Wjh1a7Gt1l63bTelbk+PR9u6Wrvc3VW33Z7u08tqZdvOkxARQfHCxXvRar3XlrqkShXFC5nJhRCSQCA3QkgIEEhCLuQ6fM8fM30SgZAMZOY3l/drrVlrhjx98n2SMW9+w/QXCQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAyTJkyxWbOnMmNGzdu3OK4SWp1/fM7682cOdMAAPGRVOL653fWI2AAED8RMPcIGADETwTMPQIGAPETAYvLckm7JVWO8vHjJC2WVCupXNKnxnNSAgYA8RMBi8sXFY3SaAGbLek1RUN2qaTC8ZyUgAFA/ETA4jZVowfsEUnXjXi8RdIZY52QgAFA/ETA4jZVowfsVUmfH/F4taRLxjohAQOQbXZur7XSBd+w2vLQUZ9DBCxuUzV6wFbq0IDNHOXYmxT94pfk5ORM4NMCAFJXZGjIwi/ebV3B08y8gFXc+cWjPpcIWNymipcQASBujTXlVjnvcjMvYOYFrHTBLNvdvPWozycCFrepGj1g39QH38RRNJ4TEjAAmWxwoN9CTwetNzjFzAvYXu8sK1n5uB2IRI7pvCJgcXlOUoukQUlNkv5d0n/EblI0XMsk1Umq0Dj+/UsiYAAyV11F2KrzZ/qrruJF/8faW1sm5NwiYO4RMACZpq+3xwoe/5kNBCebeQHb6U2zstUvTOjnEAFzj4AByCRVxW/a1rwZ/qorvOQG6+zYO+GfRwTMPQIGIBP0dHVYaNmNFglOMvMC1ph7gW0s+EvCPp8ImHsEDEC6q3h3hTXnnmfmBWwweKIVPHyr9fZ0JfRzioC5R8AApKuOtlYrvP86/+XC2rn/ZNXr303K5xYBc4+AAUhHpauett3ex8y8gPUHT7KCJ35lA/19Sfv8ImDuETAA6aS1pdFK7r7KX3VV5X/WGqrWJX0OETD3CBiAdHAgErGiFcus3fuomRewnuApFnr2DhsaHHQyjwiYewQMQKpr2VZtZfOv8Fdd5Xd+2Zq3bnY6kwiYewQMQKqKDA1Z+Pm7rDt4qpkXsH3eGVb4x8XHvA3URBABc4+AAUhFjdVltvGOzw1vvrtwtrU2N7geyycC5h4BA5BKBgf6reB3v7a+2Oa7rV6OrXvtCddjHUIEzD0CBiBV1JaHrGbuxf6qq+jeb1vHnp2uxzosETD3CBgA13r3d1vBo//lb77b4p1rG97+g+uxjkgEzD0CBsClqsK/WkPedDMvYJHgJAsv+Tfr2tfmeqwxiYC5R8AAuNDd2W7hpT/0N9/dlvcJ2xR+3fVY4yYC5h4BA5Bs5Wteth3eucOb7z7yf613f7frseIiAuYeAQOQLB17d1vRfd8ZsfnuRVZT9r7rsY6KCJh7BAxAMpS+/qS1ejlmXsD6glOs4Mk5Sd18d6KJgLlHwAAkUmvLNlu38Ep/1bXpjkutYfN612MdMxEw9wgYgEQ4EIlY0Z+WWId3hpkXsO7gqRZ+fr5FhoZcjzYhRMDcI2AAJlrz1s22Yf5X/FXXhvn/bDsa3G6+O9FEwNwjYAAmSmRoyELPzrOe4ClmXsA6vDOs6E9LU2Lz3YkmAuYeAQMwERo2r7dNd1zqr7rW3X2ltbY0uh4rYUTA3CNgAI7FQH+fhZ6YY/3Bk/zNd0tff9L1WAknAuYeAQNwtGrK3rfauRf5q67C+75rHXt3ux4rKUTA3CNgAOLV29NlBY/caoPBE828gDXnftzK31nheqykEgFzj4ABiMem8OvWmHuBv/luaNmPrLuz3fVYSScC5h4BAzAeXfvaLLzkBv/lwoa8GVZV9IbrsZwRAXOPgAEYy4a3XrKW2Oa7A8HJVvDY/7O+3h7XYzklAuYeAQMwmvbWFiu691p/1VU991NWWx5yPVZKEAFzj4ABONiBSMTW/WW57fHONvMC1hucYqGnfmODA/2uR0sZImBxmyVpi6RaSbcd5uM5kt6WtF5SuaTZY52QgAEYqbW5wUoXzvZXXRvnXW6N1WWux0o5ImBxOV5SnaRzJH1Y0gZJ0w865lFJt8TuT5fUMNZJCRgAs9jmuy/fb/u80828gHUFT7PwCwszZvPdiSYCFpfLJK0a8XhO7DbSI5J+NeL4grFOSsAANNdXWfmdX/JXXWXzr7CWxhrXY6U0EbC4XCPp8RGPr5e09KBjzpBUIalJUrukmWOdlIAB2WtocNBCz+T7m++2eWda8Z8fysjNdyeaCFhcrtWhAVty0DE/k/Tz2P3LJG2S9KHDnOsmRb/4JTk5Oa6fBwAcaNhUYlX5n/VXXSX3/C/bs3O767HShghYXMbzEuJGSWePeFwv6dQjnZQVGJBdBvr7rGD5L/3Nd3d5U6101dOux0o7ImBxOUHRIE3T8Js4Zhx0zGuSbojd/4SkHZKOO9JJCRiQPapL37G6vAuHN9994HvW0dbqeqy0JAIWt9mSqhV9N+LtsT+bK+mq2P3pktYqGrcySV8f64QEDMh8vT1dFnroFhsKTjLzAtaUe55VvPdn12OlNREw9wgYkNkq16607bnnm3kBGwpOstCDN1tPV4frsdKeCJh7BAzITJ0dey28+Af+y4Vb82bY5uLVrsfKGCJg7hEwIPOUrX7BdnrTzLyA9QcnW8HjP8/6zXcnmgiYewQMyBxtu3dY8aKr/VXXlvxLrL4y7HqsjCQC5h4BA9LfgUjESl59zPZ6Z5l5AdsfPNlCT3s2NDjoerSMJQLmHgED0tuupnorXTDLX3VVzvu8ba+pcD1WxhMBc4+AAenpQCRihS8tss7gaWZewDqDp1n4xXvYfDdJRMDcI2BA+mmqq7TKeV/wV13r7/q67dxe63qsrCIC5h4BA9LH0OCghX6fa/uDJ8c23z3Lil99lM13HRABc4+AAemhfmORbcm/xF91FS+62tp273A9VtYSAXOPgAGprb+v1wp++wvrD072N99d/+ZzrsfKeiJg7hEwIHVtWfe21ed90l91hRdfb/va97geC0bAUgIBA1LP/u5OCz14s7/57vbc/2GV77/qeiyMIALmHgEDUkvF+69YU+55w5vvPnSL7e/udD0WDiIC5h4BA1LDvvY9Fn7g+/7LhfV5F9qWdWtcj4VRiIC5R8AA99a/8azt8qYOb767/JfW39freiwcgQiYewQMcGfvriYrvud/+6uuzfmftq2bil2PhXEQAXOPgAHJdyASseJXHrY270wzL2A9wVMs9MxcNt9NIyJg7hEwILlaGmus7K6v+auuiju/aE11m1yPhTiJgLlHwIDkiAwNWfjFu60rtvnuPu90K/zDfWwDlaZEwNwjYEDiNdaUW+W8y/1VV+mCWba7eavrsXAMRMDcI2BA4gwO9Fvo6aD1BqeYeQHb451tJSuXs+rKACJg7hEwIDHqKsJWnT/TX3UV3XuNtbe2uB4LE0QEzD0CBkysvt4eCz32UxuIbb7b4p1jZW+96HosTDARMPcIGDBxqorftK15M4Y3311yg3V27HU9FhJABMw9AgYcu56uDgstu9Eisc13G3MvsI0Ff3E9FhJIBMw9AgYcm4p3V1hzbPPdweCJVvDwrdbb0+V6LCSYCJh7BAw4Oh1trVZ4/3X+y4W1c//Jqte/63osJIkImHsEDIhf6aqnbbf3sdjmuydZ6Ik5NtDf53osJJEImHsEDBi/1pZGK7n7Kn/VVZX/WWuoWud6LDggAuYeAQPGdiASsaIVy6zd++jw5rvPzmPz3SwmAuYeAQOOrGVbtZXNv8JfdZXf+WVr3rrZ9VhwTAQsbrMkbZFUK+m2UY75tqRNkjZKenasExIw4PAiQ0MWfv4u6w6eGtt89wwr/ONitoGCmRGweB0vqU7SOZI+LGmDpOkHHXOepPWSJscenzrWSQkYcKjG6jLbeMfn/FXXuoXftNaWba7HQgoRAYvLZZJWjXg8J3YbaaGkH8VzUgIGDBsc6LeC3/3a+mKb77Z6ObbutSdcj4UUJAIWl2skPT7i8fWSlh50zApFI7ZWUljRlxyPiIABUbUb1lrN3ItHbL77bevYs9P1WEhRImBxuVaHBmzJQce8KulPkv5O0jRJTZJOPMy5blL0i1+Sk5Pj+nkAONW7v9sKHv2JDQZPjG2+e65tePsPrsdCihMBi8t4XkJ8WNINIx6vlvTpI52UFRiyWVXhX60hb7qZF7BIcJKFl/ybde1rcz0W0oAIWFxOkFSv6Mrqb2/imHHQMbMk/S52/2RJ2yVNOdJJCRiyUXdnu4WX/tDffHdb3iesKrzK9VhIIyJgcZstqVrRdyPeHvuzuZKuit0/TtK9ir6NvkLSd8c6IQFDtilf87Lt8M4d3nz30Z9Y7/5u12MhzYiAuUfAkC069u62ovu+479Jo2buxVa7Ya3rsZCmRMDcI2DIBqWvP2mtXo6ZF7C+4BQrePK/2XwXx0QEzD0ChkzW2rLN1i280l91bbrjMtu2Zb3rsZABRMDcI2DIRAciESv842Lr8M4w8wLWHTzVws/Pt8jQkOvRkCFEwNwjYMg0zVs324b5X/FXXRvmX2E7Gth8FxNLBMw9AoZMERkastCz86wneIqZF7B276NWtGIZm+8iIUTA3CNgyAQNm9fbpjsuHd589+5/tdaWRtdjIYOJgLlHwJDOBvr7LPTEHOsPnjS8+e7rT7keC1lABMw9AoZ0VVP2ntXOvchfdRXe913r2Lvb9VjIEiJg7hEwpJveni4reORWf/Pd5tyPW/k7K1yPhSwjAuYeAUM62RR+3RpzL/A33w0tu9F6ujpcj4UsJALmHgFDOuja12bhJTf4Lxc25M2wqqI3XI+FLCYC5h4BQ6rb8NZL1hLbfHcgONlCj/3U+np7XI+FLCcC5h4BQ6pqb22xonuv9Vdd1XM/ZbXlIddjAWZGwFICAUOqORCJWMnK5bbHO9vMC1hvcIqFnvqNDQ70ux4N8ImAuUfAkEpamxusdME3/FXXxnmXW2NNueuxgEOIgLlHwJAKDkQiVvTy/bbPO93MC1hX8DQLv7CQzXeRskTA3CNgcK25vsrK7/ySv+oqu+ur1tJY43os4IhEwNwjYHBlaHDQQs/k+5vvtnlnWvGfH2LzXaQFETD3CBhcaNhUYlX5n/VXXSX3fMv27Nzueixg3ETA3CNgSKb+vl4rWP5Lf/PdXd5UW//XZ1yPBcRNBMw9AoZkqS59x+ryLhzefPeB71lHW6vrsYCjIgLmHgFDovX2dFnooVtsKDjJzAtYU+55VvHen12PBRwTETD3CBgSqXLtStuee76ZF7Ch4CQLPXiz7e/udD0WcMxEwNwjYEiEzo69Fl78A//lwq15/9M2F692PRYwYUTA3CNgmGhlq1+wnd40My9g/cHJVvD4z62/r9f1WMCEEgFzj4BhorTt3mHFi672V11b8i+x+sqw67GAhBABc4+A4VgdiESs+NVHrc07y8wL2P7gyRb6fa4NDQ66Hg1IGBEw9wgYjsWupnorXTDLX3VVzvuCNdVVuh4LSDgRMPcIGI7GgUjECl9aZJ3B08y8gHUGT7PClxaxDRSyhgiYewQM8Wqqq7TKeV/wV13rF/yL7Wqqdz0WkFQiYHGbJWmLpFpJtx3huGskmaRLxjohAcN4DQ0OWuj3ubY/eHJs892zrPjVR1l1ISuJgMXleEl1ks6R9GFJGyRNP8xx/yjpXUlhETBMkPqNRbYl/xJ/1VW86Gpr273D9ViAMyJgcblM0qoRj+fEbge7X9KVktaIgOEY9ff1WsFvf2H9wcnDm++++ZzrsQDnRMDico2kx0c8vl7S0oOOuVjSy7H7a0TAcAy2rHvb6vM+6a+6wot/YJ0de12PBaQEEbC4XKtDA7ZkxOMPKRqtqbHHazR6wG5S9ItfkpOT4/p5gBSzv7vTQg/e7G++uz33fKtcu9L1WEBKEQGLy1gvIU6StEdSQ+zWJ2mHxliFsQLDSBXvv2JNuecNb7770C1svgschghYXE6QVC9pmobfxDHjCMevES8hYpz2te+x8APf918urM+70KpL33E9FpCyRMDiNltStaLvRrw99mdzJV11mGPXiIBhHNa/8azt8qYOb767/JdsvguMQQTMPQKWvfbuarKSe77lr7o253/Gtm4qdj0WkBZEwNwjYNnnQCRixa88bG3emWZewHqCp1jomXw23wXiIALmHgHLLi2NNVZ219f8VVfFnV+0prpNrscC0o4ImHsELDtEhoYs/OLd1hXbfHefd7oVvXw/20ABR0kEzD0Clvkaa8qtct7l/qqrdMEs29281fVYQFoTAXOPgGWuwYF+Cz31G+sNTjHzArbHO9tKVi5n1QVMABEw9whYZqqrCFt1/kx/1VV07zXW3trieiwgY4iAuUfAMktfb4+FHvupDcQ2323xzrGyt150PRaQcUTA3CNgmaOq+E3bmjdjePPdJTdY174212MBGUkEzD0Clv56ujostOxGi8Q2323MvcA2hl5zPRaQ0UTA3CNg6a3i3RXWHNt8dzB4ohU8cqv19nS5HgvIeCJg7hGw9NTR1mqF91/nv1xYO/ciqyl7z/VYQNYQAXOPgKWf0lVP227vY7HNd0+y0BNzbKC/z/VYQFYRAXOPgKWP1pZGW3f3v/qrrqo7LrWGzetdjwVkJREw9whY6jsQiVjRimXW7n10ePPdZ+dZZGjI9WhA1hIBc4+ApbaWbdVWNv8Kf9VVfueXrXnrZtdjAVlPBMw9ApaaIkNDFn7+LusOnhrbfPcMK/zjYraBAlKECJh7BCz1bNuy3jbe8Tl/1bVu4ZXW2rLN9VgARhABc4+ApY7BgX4rePK/rS+2+W6rl2PrXnvC9VgADkMEzD0ClhpqN6y1mrkXD2++e993rGPPTtdjARiFCJh7BMyt3v3dVvDoT2wweKKZF7Ad3rlWvuZl12MBGIMImHsEzJ2qwr9aQ950My9gkeAkCy/9oXV3trseC8A4iIC5R8CSr7uz3cJLf+hvvrst7xNWFV7leiwAcRABc4+AJVf5mpdth3fu8Oa7j/6X9e7vdj0WgDiJgLlHwJKjY89OK7rvO/6bNGrmXmy1G9a6HgvAURIBc4+AJd66156wVi/HzAtYX3CKFfzu1zY40O96LADHQATMPQKWOK0t22zdwiv9VdemOy6zbVvYfBfIBCJg7hGwiXcgErHCPy62Du8MMy9g3cFTLfz8XWy+C2QQETD3CNjEat662TbM/4q/6tow/wrb0cDmu0CmEQFzj4BNjMjQkIWenWc9wVPMvIC1ex+1ohXL2HwXyFAiYO4RsGPXsHm9bbrjUn/VVXL3Vdba0uh6LAAJJALmHgE7egP9fRZ6Yo71B0/yN98tXfW067EAJIEIWNxmSdoiqVbSbYf5+M8kbZJULmm1pI+NdUICdnRqyt6z2rkX+auuwvuvs469u12PBSBJRMDicrykOknnSPqwpA2Sph90zFck/UPs/i2SXhjrpAQsPr09XVbwyK3+5rvNuR+38ndWuB4LQJKJgMXlMkmrRjyeE7uN5mJJa8c6KQEbv03h160x9wJ/893Qshutp6vD9VgAHBABi8s1kh4f8fh6SUuPcPxSSb8e66QEbGxd+9osvOQG/+XCrXkzrKroDddjAXBIBCwu1+rQgC0Z5djvSwpL+sgoH79J0S9+SU5OjuvnQUore+tFa/HOMfMCNhCcbKHHfmp9vT2uxwLgmAhYXMb7EuJXJVVJOnU8J2UFdnjtrS1WdO81/qqreu6nrK4i7HosAClCBCwuJ0iqlzRNw2/imHHQMRcr+kaP88Z7UgL2QQciEStZudz2eGebeQHrDU6x0FO/YfNdAB8gAha32ZKqFY3U7bE/myvpqtj9NyXtklQWu70y1gkJ2LDW5gYrXfANf9W1cd7l1lhT7nosAClIBMw9AhZddRW9fL/t80438wLWFTzNwi8sZPNdAKMSAXMv2wPWXF9l5Xd+yV91ld31NWtprHE9FoAUJwLmXrYGbGhw0ELP5Pub77Z5Z1rxKw+z+S6AcREBcy8bA9awqcQ2539mePPde75le3c1uR4LQBoRAXMvmwLW39drBct/6W++u8ubauv/+ozrsQCkIREw97IlYNWl71hd3oXDm+8+8D3b177H9VgA0pQImHuZHrDeni4LPXSLDQUnmXkBa8o9zyre+7PrsQCkOREw9zI5YJVrV9r23PPNvIANBSdZ6MGbbX93p+uxAGQAETD3MjFgnR17Lbz4B/7LhfV5n7Qt6952PRaADCIC5l6mBaxs9XO205tm5gWsPzjZCn77C+vv63U9FoAMIwLmXqYErG33DitedLW/6tqSf4nVbyxyPRaADCUC5l66B+xAJGLFrz5qbd5ZZl7A9gdPttDvc21ocND1aAAymAiYe+kcsF1N9Va6YJa/6qqc9wVrqqt0PRaALCAC5l46BuxAJGKFLy2yzuBpZl7AOoOnWeFLi9gGCkDSiIC5l24Ba6qrtMp5X/BXXesX/Ivtaqp3PRaALCMC5l66BGxocNBCv8+1/cGTzbyA7fXOspJXH2PVBcAJETD30iFg9RuLbEv+Jf6qq3jR1da2e4frsQBkMREw91I5YP19vdH/H1dwspkXsJ3eNCtb/ZzrsQCAgKWCVA3YlnVvW33eJ/1VV3jxD6yzY6/rsQDAzAhYSki1gO3v7rTQgzf7m+9uzz3fKteudD0WAHyACJh7qRSwivdfsabc8/zNdwse/rH19nS5HgsADiEC5l4qBGxf+x4rfOB7IzbfvdCqS99xPRYAjEoEzD3XAVv/xrO2y5sa23z3pOhvTGbzXQApTgTMPVcB27uryUru+Za/6tqc/xlr2FTiZBYAiJcImHvJDtiBSMSKX3nY2rwzzbyA9QRPsdAz+Wy+CyCtiIC5l8yAtTTWWNldX/NXXeV3fsma66uS9vkBYKKIgLmXjIBFhoYs/OLd1hXbfHefd7oVvXw/20ABSFsiYO4lOmCNNeVWOe9yf9VVuuAbtrt5a0I/JwAkmgiYe4kK2OBAv4We+o31BqeYeQHb451tJSuXs+oCkBFEwNxLRMDqKsJWnT/TX3UV3Xuttbe2TPjnAQBXRMDcm8iA9fX2WOixn9pAbPPdFu9c2/DWSxN2fgBIFSJgcZslaYukWkm3HebjH5H0QuzjhZKmjnXCiQpYVfGbtjVvxvDmu0tusK59bRNybgBINSJgcTleUp2kcyR9WNIGSdMPOubHkh6O3f+uojE7omMNWE9Xh4WW3WiR2Oa7jbkX2MbQaxP0FAGA1CQCFpfLJK0a8XhO7DbSqthxknSCpD2SjjvSSY8lYOXvrLDm2Oa7g8ETreCRW613f/cEPkUAIDWJgMXlGkmPj3h8vaSlBx1TKemsEY/rJJ18pJMeTcB693db4X3f9V8urJ17kdWUvZeApwgApCYRsLhcq0MDtuSgYzbq0IBNOcy5blL0i1+Sk5MT9zfuQCRi5Xd+2fqDJ1noiTk20N+XgKcHAKQuEbC4pNRLiDsaNlvD5vUT/JQAgPQgAhaXEyTVS5qm4TdxzDjomP/UB9/E8eJYJ3X961QAIB2JgMVttqRqRV8avD32Z3MlXRW7//eSXlL0bfRFir5j8YgIGADETwTMPQIGAPETAXOPgAFA/ETA3CNgABA/ETD3CBgAxE8EzD0CBgDxEwFzj4ABQPxEwFJCq2K7chzFreEY/rfpeuOas+PGNWfH7ViuuVVIayWuB3CAa84OXHN2yMZrRkw2fvO55uzANWeHbLxmxGTjN59rzg5cc3bIxmtGzE2uB3CAa84OXHN2yMZrBgAAAAAgQWZJ2qLoLve3HebjH5H0QuzjhZKmJm2yxBnrmn8maZOkckmrJX0seaMlzFjX/DfXSDJJlyRjqAQbzzV/W9Hv9UZJzyZprkQZ63pzJL0tab2iz+3ZyRstYZZL2q3ob6w/nOMkLVb0a1Iu6VNJmgtJcLyiv77lHA3/HrLpBx3zY33w95C9kLTpEmM81/wVSf8Qu3+LsuOaJekfJb0rKaz0D9h4rvk8RX+YT449PjVp00288Vzvo4o+nxX7WEOyhkugLyoapdECNlvSa4qG7FJF/xKODJGQ3wSd4sZzzSNdLGltQidKvPFe8/2SrpS0RukfsPFc80JJP0raRIk1nut9RNKvRhxfkIS5kmGqRg/YI5KuG/F4i6QzEj0QkuMaSY+PeHy9pKUHHVMp6awRj+sknZzguRJpPNc80lJJv07oRIk3nmu+WNLLsftrlP4BG881r1A0YmsVXXXOSs5oCTGe6z1DUoWkJkntkmYmZ7SEm6rRA/aqpM+PeLxa6f/cRsy1OvRJv+SgYzbq0IBNSfBciTSea/6b7yv6g+0jiR4qwca65g8pGq2pscdrlP7/kY/n+/yqpD9J+jtJ0xT9wX5iUqabeOO53p9J+nns/mWK/tvfhxI/WsJN1egBW6lDA5Yp4c56vIQ4+stpX5VUpfT+d5G/GeuaJyn6fW2I3fok7VB6R2w83+eHJd0w4vFqSZ9O7FgJM57r3Sjp7BGP65UZz++p4iXErHSCok/iaRr+h98ZBx3zn/rgmzheTNp0iTGea75Y0ZXmeckdLWHGc80jrVF6x0sa3zXPkvS72P2TJW1X+r66MJ7rfU3Dwf6Eon9JSee/jP7NVI0esG/qg2/iKErSTEiS2ZKqFf2BfXvsz+ZKuip2/+8lvaTo21CLFH2XU7ob65rflLRLUlns9kqyB0yAsa55pDVK/4BJY1/zcZLuVfSltApF/4KWzsa63umK/nvfBkWf119P9oAJ8JykFkmDir4E/O+S/iN2k6Lf42WKfk0qlBnPawAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADIHAj6lOkAAAAGSURBVP8fa5bx9eNL0HAAAAAASUVORK5CYII=" width="432">


```%run file.py``` -- чтобы запустить скрипт в тетрадке

```%%writefile``` -- чтобы записать содержимое ячейки в файл, например:

```
%%writefile somefile.py
def somefunction(x):
    return(x)
```

```%%latex``` -- для отображения ячеек, написанных в LaTeX


```latex
%%latex
\begin{align}
a = \frac{1}{2} && b = \frac{6}{7}
\end{align}
```


\begin{align}
a = \frac{1}{2} && b = \frac{6}{7}
\end{align}


**Находить и устранять ошибки**

```%debug``` (чтобы выйти из этого режима, нажмите ```q```)


```python
x = [1,2]
y = 3
result = x+y 
print(result)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-26-190c858d62c4> in <module>()
          1 x = [1,2]
          2 y = 3
    ----> 3 result = x+y
          4 print(result)
    

    TypeError: can only concatenate list (not "int") to list



```python
%debug
```

    > [1;32m<ipython-input-26-190c858d62c4>[0m(3)[0;36m<module>[1;34m()[0m
    [1;32m      1 [1;33m[0mx[0m [1;33m=[0m [1;33m[[0m[1;36m1[0m[1;33m,[0m[1;36m2[0m[1;33m][0m[1;33m[0m[0m
    [0m[1;32m      2 [1;33m[0my[0m [1;33m=[0m [1;36m3[0m[1;33m[0m[0m
    [0m[1;32m----> 3 [1;33m[0mresult[0m [1;33m=[0m [0mx[0m[1;33m+[0m[0my[0m[1;33m[0m[0m
    [0m[1;32m      4 [1;33m[0mprint[0m[1;33m([0m[0mresult[0m[1;33m)[0m[1;33m[0m[0m
    [0m
    ipdb> q
    

**Распечатывать красиво** -- ```pprint module```


```python
json_string = """{"organisation": "Python Software Foundation",
                 "officers": [
                            {"first_name": "Guido", "last_name":"Rossum", "position":"president"},
                            {"first_name": "Diana", "last_name":"Clarke", "position":"chair"},
                            {"first_name": "Naomi", "last_name":"Ceder", "position":"vice chair"},
                            {"first_name": "Van", "last_name":"Lindberg", "position":"vice chair"},
                            {"first_name": "Ewa", "last_name":"Jodlowska", "position":"director of operations"}
                            ],
                "type": "non-profit",
                "country": "USA",
                "founded": 2001,
                "members": 244,
                "budget": 750000,
                "url": "www.python.org/psf/"}"""
```


```python
import json

data = json.loads(json_string)
print(type(data))  # распечатаем тип объекта и убедимся, что теперь это не строка, а словарь
```

    <class 'dict'>
    


```python
from pprint import pprint

pprint(data) # посмотрим на сам этот словарь
```

    {'budget': 750000,
     'country': 'USA',
     'founded': 2001,
     'members': 244,
     'officers': [{'first_name': 'Guido',
                   'last_name': 'Rossum',
                   'position': 'president'},
                  {'first_name': 'Diana',
                   'last_name': 'Clarke',
                   'position': 'chair'},
                  {'first_name': 'Naomi',
                   'last_name': 'Ceder',
                   'position': 'vice chair'},
                  {'first_name': 'Van',
                   'last_name': 'Lindberg',
                   'position': 'vice chair'},
                  {'first_name': 'Ewa',
                   'last_name': 'Jodlowska',
                   'position': 'director of operations'}],
     'organisation': 'Python Software Foundation',
     'type': 'non-profit',
     'url': 'www.python.org/psf/'}
    


```python
print(data)#для сравнения
```

    {'organisation': 'Python Software Foundation', 'officers': [{'first_name': 'Guido', 'last_name': 'Rossum', 'position': 'president'}, {'first_name': 'Diana', 'last_name': 'Clarke', 'position': 'chair'}, {'first_name': 'Naomi', 'last_name': 'Ceder', 'position': 'vice chair'}, {'first_name': 'Van', 'last_name': 'Lindberg', 'position': 'vice chair'}, {'first_name': 'Ewa', 'last_name': 'Jodlowska', 'position': 'director of operations'}], 'type': 'non-profit', 'country': 'USA', 'founded': 2001, 'members': 244, 'budget': 750000, 'url': 'www.python.org/psf/'}
    

**Выделение комментариев цветом**

<div class="alert alert-block alert-info">
<b>Tip:</b> Use blue boxes (alert-info) for tips and notes. 
If it’s a note, you don’t have to include the word “Note”.
</div>

<div class="alert alert-block alert-warning">
<b>Example:</b> Yellow Boxes are generally used to include additional examples or mathematical formulas.
</div>

<div class="alert alert-block alert-success">
Use green box only when necessary like to display links to related content.
</div>

<div class="alert alert-block alert-danger">
It is good to avoid red boxes but can be used to alert users to not delete some important part of code etc. 
</div>

**Как распечатать результаты всех операций в ячейке**


```python
from IPython.core.interactiveshell import InteractiveShell  
InteractiveShell.ast_node_interactivity = "all"
```


```python
10+5          
11+6
12+7
```




    15






    17






    19




```python
InteractiveShell.ast_node_interactivity = "last_expr" #вернуться к исходным настройкам
```


```python
10+5          
11+6
12+7
```




    19



**Автоматическое комментирование** -- ```Ctrl/Cmd + /``` -- закомментирует выделенный участок кода; чтобы убрать комментарий, нажмите повторно. 

**Удалили код или ячейку с кодом случайно?**

+ Чтобы вернуть удалённый внутри ячейки код, нажмите ```CTRL/CMD+Z```

+ Чтобы вернуть удалённую ячейку, нажмите ```ESC+Z``` или ```EDIT > Undo Delete Cells```
