function memo=f_ntc_LpSq_ADMM_dct(obs,opts,memo)
% ADMM solver for tensor completion with Lp(Sq) regularization in the DCT domain.

lambda=opts.para.lambda;
rho=opts.para.rho;
nu=opts.para.nu;
p=opts.p;
q=opts.q;

if isfield(opts,'weightGap')
    weightGap=opts.weightGap;
else
    weightGap=1;
end
if isfield(opts,'recordGap')
    recordGap=opts.recordGap;
else
    recordGap=1;
end

normTruth=norm(double(memo.truth(:)));
B=zeros(obs.tsize); B(obs.idx)=1;
T=zeros(obs.tsize); T(obs.idx)=obs.y;

X=zeros(obs.tsize);
if isfield(opts,'initL')
    X=opts.initL;
end
Y=X;

if isfield(opts,'initL')
    [~,weightS,~]=f_tsvd_dct(X);
    weightS=f_fdiag_to_matrix(weightS);
else
    weightS=zeros(min(obs.tsize(1),obs.tsize(2)),obs.tsize(3));
end

fprintf('++++LpSq-ADMM-DCT: p=%g, q=%g, weightGap=%d++++\n',p,q,weightGap);

for iter=1:opts.MAX_ITER_OUT
    oldX=X;

    Z=(Y+rho*X+B.*T)./(rho+B);
    X_tmp=Z-Y/rho;
    [X,newS]=f_prox_t_LpSq_dct(X_tmp,lambda/rho,weightS,p,q);

    memo.iter=iter;
    memo.rho(iter)=rho;
    memo.eps(iter)=norm(double(X(:)-oldX(:)))/(norm(double(oldX(:)))+eps);

    if mod(iter,recordGap)==0 || iter==1
        memo.err(iter)=norm(double(X(:)-memo.truth(:)))/normTruth;
        memo.pnsr(iter)=h_Psnr(memo.truth(:),X(:));
    end

    if opts.verbose && mod(iter,memo.printerInterval)==0
        if memo.err(iter)==0
            tmpErr=norm(double(X(:)-memo.truth(:)))/normTruth;
            tmpPsnr=h_Psnr(memo.truth(:),X(:));
        else
            tmpErr=memo.err(iter);
            tmpPsnr=memo.pnsr(iter);
        end
        fprintf('++%d: eps=%0.2e, err=%0.2e, rho=%0.2e, PSNR=%0.2f\n', ...
            iter,memo.eps(iter),tmpErr,memo.rho(iter),tmpPsnr);
    end

    if (memo.eps(iter)<opts.MAX_EPS) && (iter>60) && memo.eps(iter)>1e-10
        memo.err(iter)=norm(double(X(:)-memo.truth(:)))/normTruth;
        memo.pnsr(iter)=h_Psnr(memo.truth(:),X(:));
        fprintf('Stopped:%d: eps=%0.2e, err=%0.2e, rho=%0.2e, PSNR=%0.2f\n', ...
            iter,memo.eps(iter),memo.err(iter),memo.rho(iter),memo.pnsr(iter));
        break;
    end

    Y=Y+rho*(X-Z);
    if mod(iter,weightGap)==0
        weightS=newS;
    end
    rho=min(rho*nu,opts.MAX_RHO);
end

if memo.err(memo.iter)==0
    memo.err(memo.iter)=norm(double(X(:)-memo.truth(:)))/normTruth;
end
if memo.pnsr(memo.iter)==0
    memo.pnsr(memo.iter)=h_Psnr(memo.truth(:),X(:));
end
memo.T_hat=X;
end
