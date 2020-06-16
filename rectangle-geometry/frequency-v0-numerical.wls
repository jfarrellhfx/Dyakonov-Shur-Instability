#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(*(* Jack Farrell, Dept. of Physics, University of Toronto*)*)
(*(* Script to try and investigate the frequency / growth rate of the quasinormal modes of viscous electrons subject to Dyakonov-Shur Boundary Conditions *)*)
(*ClearAll[v0,\[Gamma],\[Nu], x, J, n, t, vals, vals2, actualVals]*)
(*(* Parameters *)*)
(*v0s = Table[i,{i,0.01, 1.5,0.1}];*)
(*freqs = Table[0, {i, 1, Length[v0s]}];*)
(*For[i=1,i < Length[v0s]+1, i++,*)
(*\[Nu]=0.01;*)
(*\[Gamma]=0.04;*)
(*(* Equations *)*)
(*eq1=D[J[t,x],t]+D[2*v0s[[i]]*J[t,x],x]+D[n[t,x],x] - D[J[t,x],{x,2}]*\[Nu] == -\[Gamma]*J[t,x] + NeumannValue[0, x ==0];*)
(*eq2=D[n[t,x],t]+D[J[t,x],x]==0;*)
(*(* Get Eigenvalues Numerically *)*)
(*vals:=NDEigenvalues[{eq1, eq2, DirichletCondition[J[t,x]==0,x==1], DirichletCondition[n[t,x]==0, x==0]}, {J[t,x], n[t,x]}, t, {x, 0, 1}, 10];*)
(*vals2 = Table[I*vals[[i]], {i, 1, Length[vals]}]; *)
(*actualVals = Sort[vals2,Abs[Abs[Re[#1]-Pi/2]] < Abs[Abs[Re[#2]]-Pi/2] &]; (* We want the ones with real part close to Pi / 2 *)*)
(*freqs[[i]]=Abs[Re[actualVals[[1]]]]*)
(*]*)


(* ::Input:: *)
(*freqs*)


(* ::Input:: *)
(*ListPlot[Thread[{v0s,freqs}]]*)
