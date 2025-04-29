import { createParamDecorator, ExecutionContext } from '@nestjs/common';

export const CurrentUser = createParamDecorator(
  (data: string | undefined, ctx: ExecutionContext) => {
    const request = ctx.switchToHttp().getRequest();
    const user = request.user;
    console.log('data CurrentUser', data);
    if (data) {
      return user?.[data];
    }
    console.log('user CurrentUser', user);

    return user;
  },
);
