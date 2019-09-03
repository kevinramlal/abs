%
% This function calculates the gradient of the log-likelihood for the 
% proportional hazard model using the log-logistics baseline distribution
%
function [grad] = log_log_grad(param, tb, te, event, covars)

g = param(1);         % Amplitude of the baseline hazard; gamma in the notation
p = param(2);         % Shape of baseline hazard; p in the notation
coef = param(3:end);  % Coefficients for covariates; beta in the notation

dlldg1 = sum(event.*(p/g - (p*g^(p-1)*(te.^p))./(1+(g*te).^p)));
if ~isempty(covars)
    dlldg2 = sum(((p*g^(p-1)).*((te.^p)./(1+(g*te).^p) - (tb.^p)./(1+(g*tb).^p))).*exp(covars*coef));
else
    dlldg2 = sum(((p*g^(p-1)).*((te.^p)./(1+(g*te).^p) - (tb.^p)./(1+(g*tb).^p))));
end
dlldg  = -(dlldg1 - dlldg2);
      
dlldp1 = sum(event.*(1/p + log(g*te) - ((g*te).^p).*log(g*te)./(1+(g*te).^p)));
% When tb = 0, calculate the derivative of the unconditional survival function. 
% This is because the derivative of the conditional survival function does not
% generalize to the unconditional case when tb = 0. There is a singularity on
% log(g*tb) for tb = 0.
warning off;
ln_gtb = log(g*tb);
ln_gtb(isinf(ln_gtb)) = 0;
warning on;

if ~isempty(covars)
    dlldp2 = sum((((g*te).^p).*log(g*te)./(1+(g*te).^p) - ((g*tb).^p).*ln_gtb./(1+(g*tb).^p)).*exp(covars*coef));      
else
    dlldp2 = sum(((g*te).^p).*log(g*te)./(1+(g*te).^p) - ((g*tb).^p).*ln_gtb./(1+(g*tb).^p));      
end    
dlldp = -(dlldp1 - dlldp2);

grad = [dlldg; dlldp];

for i=1:length(coef)
    dlldc1 = sum(event.*covars(:,i));
    dlldc2 = sum((log(1+(g*te).^p) - log(1+(g*tb).^p)).*exp(covars*coef).*covars(:,i));    
    dlldc  = -(dlldc1 - dlldc2);
    
    grad = [grad; dlldc];
end

end
