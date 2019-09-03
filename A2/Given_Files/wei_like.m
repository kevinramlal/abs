% This function calculates the log likelihood for an exponential hazard
% model with Weibull baseline hazard.  It can be used to solve for
% the parameters of the model.

function [logL, grad] = wei_like(param, tb, te, event, covars)

global phist cnt;

% Get the number of parameters
nparams  = length(param);
nentries = length(te);

g = param(1);         % Amplitude of the baseline hazard; gamma in the notation
p = param(2);         % Shape of baseline hazard; p in the notation
coef = param(3:end);  % Coefficients for covariates; beta in the notation

% The following variables are vectors with a row for each episode
% Log of baseline hazard
logh = (log(p) + log(g) + (p-1)*(log(g)+log(te))); 

logc = zeros(nentries, 1);
logF = -((g*te).^p - (g*tb).^p);
if ~isempty(covars)
    % Product of covarites and coefficients 
    logc = covars*coef;
    % Log of conditional survival function
    logF = logF.*exp(covars*coef);
end

% Construct the negative of log likelihood
logL = -(sum(event.*(logh+logc)) + sum(logF));

% Calculate the derivative of the log likelihood with respect to each parameter.
% In order for the maximum likelihood estimation to converge it is necessary to
% provide these derivatives so that the search algogrithm knows which direction
% to search in.

[grad] = z_wei_grad(param, tb, te, event, covars);

% matrix phist keeps track of parameter convergence history
if rem(cnt, nparams+1) == 0
    phist = [phist; param'];
end
cnt = cnt+1;

end
