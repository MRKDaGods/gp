"use client";

import { FormEvent, useEffect, useState } from 'react';
import { AlertTriangle, ExternalLink } from 'lucide-react';

import { KaggleCredentialsStatus } from '@/components/layout/kaggle-status-indicator';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useKaggleCredentialsStore } from '@/lib/kaggle-credentials-store';

export interface KaggleCredentialsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function KaggleCredentialsModal({ open, onOpenChange }: KaggleCredentialsModalProps) {
  const credentials = useKaggleCredentialsStore((s) => s.credentials);
  const setCredentials = useKaggleCredentialsStore((s) => s.setCredentials);
  const clearCredentials = useKaggleCredentialsStore((s) => s.clearCredentials);
  const [username, setUsername] = useState('');
  const [key, setKey] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;

    setUsername(credentials?.username ?? '');
    setKey(credentials?.key ?? '');
    setError(null);
  }, [credentials, open]);

  const handleSave = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const trimmedUsername = username.trim();
    const trimmedKey = key.trim();

    if (!trimmedUsername || !trimmedKey) {
      setError('Username and API key are required.');
      return;
    }

    setCredentials({ username: trimmedUsername, key: trimmedKey });
    setError(null);
    onOpenChange(false);
  };

  const handleClear = () => {
    clearCredentials();
    setUsername('');
    setKey('');
    setError(null);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[520px]">
        <DialogHeader>
          <DialogTitle>Kaggle Credentials</DialogTitle>
          <DialogDescription>
            Configure the Kaggle account used for optional offloaded pipeline stages.
          </DialogDescription>
        </DialogHeader>

        <div className="rounded-md border border-yellow-600/30 bg-yellow-600/10 p-3 text-sm text-yellow-900 dark:text-yellow-100">
          <div className="flex gap-2">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-600 dark:text-yellow-400" />
            <p>
              These credentials are stored in your browser&apos;s localStorage and sent to the backend per request.
              They are NEVER written to the backend&apos;s disk. If you don&apos;t provide credentials, the backend uses the
              server-configured account.
            </p>
          </div>
        </div>

        <form className="space-y-4" onSubmit={handleSave}>
          <div className="rounded-md border bg-muted/30 px-3 py-2 text-sm">
            <span className="mr-2 text-muted-foreground">Current status:</span>
            <KaggleCredentialsStatus />
          </div>

          <div className="space-y-2">
            <Label htmlFor="kaggle-username">Username</Label>
            <Input
              id="kaggle-username"
              value={username}
              onChange={(event) => {
                setUsername(event.target.value);
                if (error) setError(null);
              }}
              placeholder="kaggle-user"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="kaggle-api-key">API key</Label>
            <Input
              id="kaggle-api-key"
              type="password"
              value={key}
              onChange={(event) => {
                setKey(event.target.value);
                if (error) setError(null);
              }}
              placeholder="Paste your Kaggle API key"
              required
            />
          </div>

          {error && <p className="text-sm font-medium text-destructive">{error}</p>}

          <p className="text-sm text-muted-foreground">
            Get your API key from{' '}
            <a
              href="https://www.kaggle.com/settings/account"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 font-medium text-primary hover:underline"
            >
              kaggle.com/&lt;username&gt;/account
              <ExternalLink className="h-3.5 w-3.5" />
            </a>{' '}
            {'→'} Create New Token
          </p>

          <DialogFooter className="gap-2 sm:gap-0">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="button" variant="outline" onClick={handleClear} disabled={!credentials}>
              Clear
            </Button>
            <Button type="submit">Save</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
